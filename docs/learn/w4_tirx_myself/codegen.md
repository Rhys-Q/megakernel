# TIRX Codegen 原理与实现梳理

本文基于当前仓库 `3rdparty/tvm` 的源码，整理 `tvm.tirx` 从 lowered IR 到
runtime module 的 codegen 主线。

如果前面已经学过 TIRX 前端、IR 和 pass，可以先把 codegen 理解成这句话：

```text
TIRX codegen 不负责理解高层 DSL 语义。
它消费已经被 lowering pipeline 规范化后的 tirx.PrimFunc，
按 target 拆成 host/device module，再调用 target.build.<kind> 生成 runtime.Module。
```

这里有两层要分清：

| 层次 | 入口 | 主要职责 |
| --- | --- | --- |
| build orchestration | `python/tvm/tirx/build.py` | 绑定 target、选择 pipeline、拆 host/device、跑 finalization、分发到 `target.build.*`。 |
| target codegen | `src/target/*` | 把 `tirx.PrimFunc` 翻译成 C/CUDA/OpenCL/LLVM IR/NKI/SPIR-V 等目标代码，并包装成 `runtime.Module`。 |

## 1. 总流程

最常见入口是：

```text
tvm.compile(mod, target)
  -> tvm.tirx.build(mod, target, pipeline="default")
```

也可以直接调用：

```python
from tvm import tirx

rt_mod = tirx.build(mod, target="cuda", pipeline="tirx")
```

源码入口：

| 入口 | 位置 | 说明 |
| --- | --- | --- |
| `tvm.tirx.build` | `3rdparty/tvm/python/tvm/tirx/build.py` | TIRX build 主入口。 |
| `tirx.build` global func | `build.py` 末尾 | `tvm.register_global_func("tirx.build", build)`。 |
| `tvm.build` | `python/tvm/driver/build_module.py` | 已 deprecated，直接 warning 后转调 `tvm.tirx.build`。 |
| `tvm.compile` | `python/tvm/driver/build_module.py` | 非 Relax module 走 `tvm.tirx.build`，再包成 `runtime.Executable`。 |
| Relax VM build | `python/tvm/relax/vm_build.py` | Relax lowering 出 TIR module 后调用 `tvm.tirx.build`。 |

`tirx.build` 的主线可以画成：

```text
PrimFunc / IRModule
  -> PrimFunc 包成 IRModule
  -> 决定 target_to_bind / target / target_host
  -> BindTarget(target_to_bind.with_host(target_host))
  -> 选择并执行 TIR pipeline
  -> split_host_device_mods
  -> finalize_host_passes / finalize_device_passes
  -> codegen_build(device_mod, device_target)
  -> codegen_build(host_mod, target_host)
  -> host runtime module import device runtime modules
  -> 返回 runtime.Module
```

对应代码点：

| 步骤 | 代码 |
| --- | --- |
| target 和 host target 决策 | `build.py::build` Step 0-2 |
| target 绑定 | `tirx.transform.BindTarget` |
| pipeline 选择 | `tirx.get_tir_pipeline` / `tirx.get_default_tir_pipeline` |
| host/device module 拆分 | `build.py::split_host_device_mods` |
| codegen 分发 | `build.py::codegen_build` |
| runtime module 合并 | `build.py::tir_to_runtime` |

## 2. Codegen 前的 IR 契约

target codegen 不是从 `TilePrimitiveCall`、`SBlock`、layout DSL 直接发码。
这些高层结构要先被 pipeline 降低。

以 `pipeline="tirx"` 为例，主 pipeline 在
`python/tvm/tirx/compilation_pipeline.py::tirx_pipeline`：

```text
LowerTIRx
  -> TilePrimitiveDispatch
  -> LowerTIRxCleanup
  -> LowerTIRxStripExecScope
UnifyThreadBinding
Simplify
LowerTIRxOpaque
FlattenBuffer
BF16ComputeLegalize
NarrowDataType(32)
VectorizeLoop
UnrollLoop
Simplify
CommonSubexprElim
FP8ComputeLegalize
VerifyMemory
AnnotateEntryFunc
AnnotateDeviceRegions
SplitHostDevice
MakePackedAPI
FP8StorageLegalize
BF16StorageLegalize
LowerDeviceKernelLaunch
```

进入 target codegen 时，IR 通常应该满足：

| 要求 | 原因 |
| --- | --- |
| module 里都是 `tirx.PrimFunc` | 各 `target.build.*` 都检查 `PrimFuncNode`。 |
| `PrimFunc.attrs["target"]` 已存在 | host/device 分组和 intrinsic lowering 都依赖 target。 |
| buffer 访问基本已 flatten | `CodeGenC` 的 `BufferLoad/BufferStore` 要求一维 index。 |
| `TilePrimitiveCall` 已消失 | tile primitive 已在 `LowerTIRx` 中 dispatch 成低层语句或 intrinsic call。 |
| thread binding 表示成 `AttrStmt(thread_extent)` | CUDA 等后端用它映射 `threadIdx.x`、`blockIdx.x` 并收集 launch 参数。 |
| host entry 已 packed ABI 化 | `MakePackedAPI` 生成 packed function 入口、参数 unpack、shape/stride 检查。 |
| host 调 device kernel 已降低成 runtime packed call | `LowerDeviceKernelLaunch` 生成 `tvm_call_packed` 和 launch 参数。 |
| target-specific intrinsic 已降低 | finalization 里的 `LowerIntrin` 把 target op 改写到后端可打印形态。 |

`ExecScopeStmt` 在 `CodeGenC` 和 `CodeGenLLVM` 里仍有兜底 visitor，但只是继续打印
body。正常路径下它应该在 `LowerTIRxStripExecScope` 后基本不再承载语义。

## 3. Host/Device 如何拆开

TIRX codegen 最关键的设计是：先在 IRModule 中把 host/device 函数明确拆出来，
再分别发码。

### 3.1 AnnotateDeviceRegions

位置：`src/tirx/transform/annotate_device_regions.cc`

它看函数 target 是否带 host。如果有 host，就把 device-only 区域包上
`AttrStmt(target=device_target)`：

```text
AttrStmt(thread_extent, ...)
  body

变成：

AttrStmt(target=cuda_without_host, 0)
  AttrStmt(thread_extent, ...)
    body
```

触发条件主要是：

| attr | 含义 |
| --- | --- |
| `tirx::attr::thread_extent` | 有 thread binding，必然是 device region。 |
| `tirx::attr::device_scope` | 显式 device scope。 |

如果原本已经有 `AttrStmt(target=...)`，它会保留，不重复包。

### 3.2 SplitHostDevice

位置：`src/tirx/transform/split_host_device.cc`

`SplitHostDevice` 找到 host 函数体里的 `AttrStmt(target=...)`，把这块 body 抽成新的
device `PrimFunc`：

```text
host PrimFunc
  AttrStmt(target=cuda)
    device body

变成：

host PrimFunc
  Evaluate(Call(cuda_kernel_gvar, captured_args))

cuda_kernel PrimFunc
  device body
```

新 device function 的参数来自 device body 中的 undefined vars。对非 `trn` target，
参数会按“handle 优先、再按变量名”排序；`trn` 会尽量保留原始 buffer 参数顺序。

新 device function 会带这些关键 attr：

| attr | 作用 |
| --- | --- |
| `target` | device target，例如 `cuda`。 |
| `tirx.noalias` | 后端打印 restrict 等信息。 |
| `tirx.is_global_func` | 标记为全局可见函数。 |
| `tirx.is_stir`、`tirx.num_inputs` 等 | 从原函数透传部分元信息。 |

对能传播错误码的 CPU/ext_dev/hexagon 目标，device func 可返回 `int32` 状态码；
对 CUDA 这类 device kernel，返回类型是 `void`。

### 3.3 LowerDeviceKernelLaunch

位置：`src/tirx/transform/lower_device_kernel_launch.cc`

这个 pass 做两件事。

第一，收集每个 device function 的 kernel launch 信息：

| 信息 | 来源 |
| --- | --- |
| `target` | `PrimFunc.attrs["target"].WithoutHost()` |
| `global_symbol` | `global_symbol` attr 或 GlobalVar 名 |
| `params` | device function 参数 |
| `launch_params` | `thread_extent` 的 thread tag，例如 `blockIdx.x`、`threadIdx.x` |
| `launch_args` | 对应 thread extent 表达式 |
| dynamic shared memory | shared `.dyn` allocation 的字节数 |

第二，重写 call site：

| caller/callee 关系 | 改写 |
| --- | --- |
| same target | 保留 `Call(GlobalVar, args)`，交给 codegen 当内部函数调用处理。 |
| same device type but different target | 改成 `call_extern`。 |
| host 调 device | 改成 `tvm_call_packed(global_symbol, args..., launch_args...)`。 |

同时它会更新被 host launch 的 device function：

```text
calling_conv = kDeviceKernelLaunch
tirx.kernel_launch_params = [...]
global_symbol = ...
ret_type = void
```

这一步是 host/device ABI 的连接点：host module 里留下 runtime packed call；
device module 里的 kernel function 则被标成后端需要导出的 kernel。

### 3.4 split_host_device_mods

位置：`python/tvm/tirx/build.py::split_host_device_mods`

pipeline 后，host/device 函数仍在同一个 `IRModule`。Python build 再按 target
把它们拆成：

```text
host_mod: target.kind.name in ["llvm", "c"]
device_mod_dict: Dict[Target, IRModule]
```

当前实现按 target 字符串分组，因为源码注释里说 target hash 还不可靠。

### 3.5 finalization 和 runtime module 合并

位置：`python/tvm/tirx/build.py::tir_to_runtime`

build 会先编译 device modules，再编译 host module：

```text
for each device_mod:
  dev_runtime = codegen_build(device_mod, device_target)

host_runtime = codegen_build(host_mod, target_host)

for dev_runtime in device_runtimes:
  host_runtime.import_module(dev_runtime)
```

最终返回的是 host runtime module，device code 作为 import module 挂进去。

### 3.6 host module 为什么要 import device module

host/device 分别 codegen 完之后，它们已经不是 `IRModule`，而是 runtime 层的
`ffi::Module`：

```text
host_mod   --target.build.llvm/c-->  host_runtime: ffi::Module
device_mod --target.build.cuda---->  dev_runtime:  ffi::Module
```

但是用户拿到的只能是一个入口 module。TIRX 的组装策略是：返回 host module，并把所有
device runtime module 挂到 host module 的 imports 上：

```python
mhost = codegen_build(mhost_all, target_host)
for dev_mod in device_modules:
    if dev_mod is not None:
        mhost.import_module(dev_mod)
return mhost
```

这一步解决两个问题。

第一，host wrapper 里已经没有直接的 CUDA C++ kernel launch 语法。经过
`LowerDeviceKernelLaunch` 后，host IR 里留下的是 runtime packed call：

```python
T.call_packed("main_kernel", A_ptr, 6, 2)
```

host codegen 只负责把这个 packed call 降成 TVM runtime 调用。真正知道
`main_kernel` 是 CUDA kernel、参数如何打包、grid/block 如何发射的是 CUDA
`ffi::Module`。因此运行时查找 `"main_kernel"` 时必须能从 host module 走到
imported CUDA module。

第二，host 和 device 的生命周期、导出、序列化要作为一个整体管理。`ffi::Module`
本身有 import tree：

| 操作 | 实现位置 | 作用 |
| --- | --- | --- |
| `ImportModule` | `3rdparty/tvm/3rdparty/tvm-ffi/src/ffi/extra/module.cc` | 把 child module 放进 `imports_`，并检查不会形成循环依赖。 |
| `GetFunction(name, query_imports=True)` | 同上 | 先查当前 module，查不到时递归查 imported modules。 |
| `ModuleSerializer` | `3rdparty/tvm/src/target/codegen.cc` | 导出/序列化时保存 import tree，并按拓扑顺序恢复 imports。 |

所以返回的 `rt_mod` 可以理解成：

```text
host_runtime_module
  imports[0] = cuda_runtime_module
  imports[1] = 其他 device/runtime module
```

运行时路径是：

```text
用户调用 host entry，例如 f(a)
  -> host packed wrapper 解包 DLTensor / 检查 dtype shape device
  -> host wrapper 调 tvm_call_packed("main_kernel", A_ptr, grid, block)
  -> runtime 按 query_imports 查找 "main_kernel"
  -> 在 imported CUDA module 中找到 CUDAWrappedFunc
  -> CUDAWrappedFunc 根据 FunctionInfo 解析 launch 参数
  -> cuModuleLoadData / cuModuleGetFunction / cuLaunchKernelEx
```

这里的关键点是：`import_module` 不是把 CUDA source 文本拼进 host source，也不是把两个
`IRModule` 重新合并；它是在 runtime module 层建立父子关系。host module 仍然是 host
target 的产物，CUDA module 仍然是 CUDA target 的产物，只是 host module 的 imports
让 runtime 能跨 module 找到 device kernel。

## 4. target.build 分发机制

`tirx.build` 最后不会自己生成 CUDA 或 LLVM IR，而是查全局函数：

```python
build_f_name = "target.build." + target.kind.name
bf = tvm.get_global_func(build_f_name)
return bf(mod, target)
```

位置：`python/tvm/tirx/build.py::codegen_build`

C++ 侧还有同样的通用入口：

```cpp
ffi::Module Build(IRModule mod, Target target) {
  std::string build_f_name = "target.build." + target->kind->name;
  auto bf = tvm::ffi::Function::GetGlobal(build_f_name);
  return (*bf)(mod, target).cast<ffi::Module>();
}
```

位置：`src/target/codegen.cc::Build`

常见注册点：

| target kind | global func | 实现位置 |
| --- | --- | --- |
| `llvm` | `target.build.llvm` | `src/target/llvm/llvm_module.cc` |
| `c` | `target.build.c` | `src/target/source/codegen_c_host.cc` |
| `cuda` | `target.build.cuda` | `src/target/cuda/codegen_cuda.cc` |
| `nvptx` | `target.build.nvptx` | `src/target/cuda/llvm/codegen_nvptx.cc` |
| `rocm` | `target.build.rocm` | `src/target/rocm/llvm/codegen_amdgpu.cc` |
| `opencl` | `target.build.opencl` | `src/target/opencl/codegen_opencl.cc` |
| `vulkan` | `target.build.vulkan` | `src/target/vulkan/build_vulkan.cc` |
| `metal` | `target.build.metal` | `src/target/metal/codegen_metal.cc` |
| `webgpu` | `target.build.webgpu` | `src/target/webgpu/codegen_webgpu.cc` |
| `trn` | `target.build.trn` | `src/target/source/codegen_trn.cc` |

所以新增 target 后端的最小接口就是注册一个 `target.build.<kind>`，输入
`IRModule + Target`，输出 `ffi::Module`。

## 5. C-style source codegen 基类

很多 source backend 复用 `CodeGenC`：

```text
CodeGenSourceBase
  -> CodeGenC
      -> CodeGenCHost
      -> CodeGenCUDA
      -> CodeGenOpenCL
      -> CodeGenMetal
      -> CodeGenWebGPU
      -> CodeGenTrainium
```

核心文件：

| 文件 | 作用 |
| --- | --- |
| `src/target/source/codegen_source_base.h` | 缩进、变量命名、SSA 临时变量、stream 管理。 |
| `src/target/source/codegen_c.h` | C-style backend 的 visitor 接口。 |
| `src/target/source/codegen_c.cc` | 大多数 `tirx.PrimExpr` / `tirx.Stmt` 的默认打印逻辑。 |

`CodeGenC` 同时继承：

```cpp
ExprFunctor<void(const PrimExpr&, std::ostream&)>
StmtFunctor<void(const Stmt&)>
CodeGenSourceBase
```

也就是说它直接访问 TIRX IR，不是先转成老 `tir::Stmt`。

### 5.1 CodeGenC 发码流程

C-style codegen 的入口虽然分散在各个 `target.build.*`，但内部流程基本一致：

```text
BuildXXX(IRModule mod, Target target)
  -> CodeGenXXX cg(target)
  -> cg.Init(...)
  -> 遍历 mod->functions，检查每个 BaseFunc 都是 tirx.PrimFunc
  -> cg.DeclareFunction(gvar, prim_func)
  -> cg.AddFunction(gvar, prim_func)
  -> code = cg.Finish()
  -> SourceModule/CUDAModule/DeviceSourceModuleCreate(...)
```

以 CUDA 为例，`src/target/cuda/codegen_cuda.cc::BuildCUDA` 做的是：

```cpp
ffi::Module BuildCUDA(IRModule mod, Target target) {
  CodeGenCUDA cg(target);
  cg.Init(/*output_ssa=*/false);

  ffi::Map<GlobalVar, PrimFunc> functions;
  for (auto [gvar, base_func] : mod->functions) {
    TVM_FFI_ICHECK(base_func->IsInstance<PrimFuncNode>());
    auto prim_func = Downcast<PrimFunc>(base_func);
    // CUDA 只接受 device kernel 或 device helper。
    auto calling_conv = prim_func->GetAttr<Integer>(
        tvm::attr::kCallingConv, Integer(tvm::CallingConv::kDefault));
    TVM_FFI_ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch ||
                   calling_conv == CallingConv::kDefault);
    functions.Set(gvar, prim_func);
  }

  for (auto [gvar, prim_func] : functions) {
    cg.DeclareFunction(gvar, prim_func);
  }
  for (auto [gvar, prim_func] : functions) {
    cg.AddFunction(gvar, prim_func);
  }

  std::string code = cg.Finish();
  return CUDAModuleCreateWithFallback(...);
}
```

为什么先 `DeclareFunction` 再 `AddFunction`？因为一个 `IRModule` 里可能有多个 device
helper 函数，函数体里也可能调用另一个 `GlobalVar`。第一轮先给所有函数分配稳定的
最终符号并打印 forward declaration，第二轮打印函数体时，`Call(GlobalVar, ...)`
才能通过 `GetFunctionName(gvar)` 找到被调函数名。

`AddFunction(gvar, func)` 是真正遍历 `PrimFunc` 的地方。`CodeGenC::AddFunction`
的大致结构是：

```text
AddFunction(gvar, func)
  -> DeclareFunction(gvar, func)  # 如果已经声明过就是 no-op
  -> InitFuncState(func)
       清空当前函数级状态
       保留 C/CUDA 关键字
       清空 var_idmap / handle dtype / storage scope 等
  -> PrintFunctionSignature(function_name, func, stream)
       PrintFuncPrefix
       PrintType(func->ret_type)
       PrintExtraAttrs
       遍历 func->params，给每个 tirx.Var 分配 C 变量名
  -> stream << " {"
  -> PreFunctionBody(func)
  -> BeginScope()
  -> PrintStmt(func->body)
       VisitStmt(func->body)
       递归遍历 Stmt tree
       语句中遇到表达式时调用 PrintExpr(expr)
       PrintExpr(expr) 再递归 VisitExpr(expr)
  -> EndScope()
  -> stream << "}"
```

因此从 `PrimFunc` 到 C/CUDA source 的核心递归关系是：

```text
PrimFunc
  params  -> PrintFunctionSignature -> AllocVarID(var)
  body    -> PrintStmt
              -> VisitStmt_(SeqStmt/For/If/BufferStore/...)
                   -> PrintExpr
                       -> VisitExpr_(Add/Call/BufferLoad/Var/...)
```

几个状态表贯穿整个函数体遍历：

| 状态 | 来源 | 用途 |
| --- | --- | --- |
| `var_idmap_` | `AllocVarID` / `BindThreadIndex` | TIRX `VarNode*` 到 C/CUDA 变量名的映射。 |
| `name_supply_` | `CodeGenSourceBase` | 避免变量名和关键字、已有变量冲突。 |
| `handle_data_type_` | 函数参数 pointer type、buffer load/store | 知道 handle 实际指向什么 dtype，减少重复 cast。 |
| `alloc_storage_scope_` | `AllocBuffer` / pointer storage scope | 打印 `__shared__`、local/global 等 storage。 |
| `decl_stream` | header、helper、pragma import | 放在最终源码最前面。 |
| `fwd_decl_stream` | `DeclareFunction` | 放 forward declarations。 |
| `stream` | `AddFunction` 主体 | 放函数定义。 |

遍历时最重要的两个入口：

```cpp
void CodeGenC::PrintStmt(const Stmt& n) {
  VisitStmt(n);
}

void CodeGenC::PrintExpr(const PrimExpr& n, std::ostream& os) {
  if (print_ssa_form_) {
    // 先打印表达式，再抽成 SSA 临时变量。
  } else {
    VisitExpr(n, os);
  }
}
```

举本节前面的 `main_kernel` 为例，CUDA codegen 的遍历顺序可以简化为：

```text
AddFunction(main_kernel)
  -> PrintFunctionSignature
       calling_conv=2 -> CodeGenCUDA 打印 extern "C" __global__
       threadIdx.x extent=2 -> PrintExtraAttrs 打印 __launch_bounds__(2)
       A_ptr -> uint64_t* __restrict__ A_ptr
  -> PrintStmt(body)
       AttrStmt(thread_extent blockIdx.x)
         -> CodeGenCUDA::BindThreadIndex(blockIdx.x)
       AttrStmt(thread_extent threadIdx.x)
         -> CodeGenCUDA::BindThreadIndex(threadIdx.x)
       BufferStore(A[blockIdx.x * 2 + threadIdx.x], value)
         -> PrintExpr(index)
              Var(blockIdx.x) -> "blockIdx.x"
              Var(threadIdx.x) -> "threadIdx.x"
         -> PrintExpr(value)
              large_uint_imm -> uint64 constant
              Add -> "(const + 3)"
         -> stream << "A_ptr[index] = value;"
  -> Finish
       header_generator(tags)
       decl_stream + fwd_decl_stream + stream
```

这就是为什么最终 CUDA source 会长成：

```cuda
extern "C" __global__ void __launch_bounds__(2) main_kernel(
    uint64_t* __restrict__ A_ptr
) {
  A_ptr[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] =
      ((uint64_t)9223372036854775931 + (uint64_t)3);
}
```

### 5.2 函数生成

函数生成分成两个阶段：先声明所有函数，再生成所有函数体。这个设计服务于
`Call(GlobalVar, ...)`，因为函数体里可能调用同一个 `IRModule` 里的另一个
`PrimFunc`。只有所有 `GlobalVar` 都先映射到最终符号名，函数体发码时才能稳定地
打印内部调用。

关键方法先列出来：

| 方法 | 作用 |
| --- | --- |
| `DeclareFunction(gvar, func)` | 分配函数名，生成 forward declaration。 |
| `AddFunction(gvar, func)` | 打印函数签名、函数体。 |
| `PrintFunctionSignature` | 打印返回类型、函数名、参数列表。 |
| `PrintFuncPrefix` | 由子类添加前缀，例如 CUDA 的 `extern "C" __global__`。 |
| `PrintExtraAttrs` | 由子类添加额外属性，例如 CUDA 的 `__launch_bounds__`。 |
| `PreFunctionBody` | 子类可在函数体最前插入语句。 |
| `Finish` | 拼接 `decl_stream + fwd_decl_stream + stream`。 |

#### 5.2.1 DeclareFunction：确定函数符号并打印前置声明

`CodeGenC::DeclareFunction(gvar, func)` 做的是“登记这个函数”，不是生成函数体。

流程：

```text
DeclareFunction(gvar, func)
  -> 如果 gvar 已经在 internal_functions_ 里，直接返回
  -> 决定 function_name
       如果 func.attrs 有 global_symbol:
         使用 global_symbol，并检查名字没有重复
       否则:
         使用 gvar->name_hint
  -> 如果 function_name == tvm_ffi_main，记录 has_tvm_ffi_main_func_
  -> internal_functions_[gvar] = function_name
  -> InitFuncState(func)
  -> PrintFunctionSignature(function_name, func, fwd_decl_stream)
  -> fwd_decl_stream << ";"
```

这里有两个重要点。

第一，最终函数名优先来自 `global_symbol`。比如 device kernel 经过
`LowerDeviceKernelLaunch` 后会带：

```text
global_symbol = "main_kernel"
calling_conv = kDeviceKernelLaunch
```

CUDA codegen 就会把它打印成：

```cuda
extern "C" __global__ void __launch_bounds__(2) main_kernel(...);
```

第二，`DeclareFunction` 会调用一次 `InitFuncState(func)`，因为打印函数签名也需要
分配参数变量名、注册 handle dtype 等函数级状态。之后 `AddFunction` 打印函数体前
会再次 `InitFuncState(func)`，保证函数定义和前置声明使用一致的状态初始化逻辑，
同时不把上一个函数体的局部变量状态带进来。

`internal_functions_` 是后续内部调用的名称表：

```text
GlobalVar(main_kernel) -> "main_kernel"
GlobalVar(helper)      -> "helper"
```

当函数体里遇到 `Call(GlobalVar, args)`，`CodeGenC::VisitExpr_(CallNode)` 会通过
`GetFunctionName(gvar)` 查这个表，然后打印普通 C-style 函数调用。

#### 5.2.2 PrintFunctionSignature：打印函数签名并绑定参数变量

`PrintFunctionSignature(function_name, func, os)` 负责打印：

```text
[backend prefix] [return type] [backend attrs] function_name(param0, param1, ...)
```

默认实现大致是：

```text
PrintFunctionSignature
  -> PrintFuncPrefix(os)
  -> PrintType(func->ret_type, os)
  -> PrintExtraAttrs(func, os)
  -> os << " " << function_name << "("
  -> 遍历 func->params
       v = func->params[i]
       如果 v 在 alloc_storage_scope_ 中，先打印 storage scope
       如果 v 是 TensorMap pointer，打印 CUDA tensor map 特殊类型
       否则 PrintType(GetType(v), os)
       如果 func 有 tirx.noalias 且 v 是 handle，打印 restrict
       os << " " << AllocVarID(v.get())
  -> os << ")"
  -> 遍历 params，把 pointer element dtype 注册到 handle_data_type_
```

子类主要覆盖这几个钩子：

| 钩子 | 默认行为 | CUDA 行为 |
| --- | --- | --- |
| `PrintFuncPrefix` | 空 | 由 `PrintFunctionSignature` 直接根据 calling conv 打印 `extern "C" __global__` 或 `extern "C" __device__`。 |
| `PrintExtraAttrs` | 空 | 从 thread extent 推导 `__launch_bounds__(...)`。 |
| `PrintType` | C 基础类型 | CUDA half/bfloat16/fp8/vector/fragment 等类型。 |
| `PrintRestrict` | 后端默认 restrict keyword | CUDA 使用 `__restrict__`。 |

以本例 `main_kernel(A_ptr)` 为例，签名生成步骤是：

```text
func.attrs["calling_conv"] = 2
  -> CUDA 打印 extern "C" __global__

func.ret_type = void
  -> PrintType(void)

func.body 里 threadIdx.x extent = 2
  -> PrintExtraAttrs 打印 __launch_bounds__(2)

param A_ptr: handle("uint64", "global")
  -> PrintType -> uint64_t*
  -> tirx.noalias=True -> __restrict__
  -> AllocVarID(A_ptr) -> A_ptr
```

最终得到：

```cuda
extern "C" __global__ void __launch_bounds__(2) main_kernel(
    uint64_t* __restrict__ A_ptr
)
```

host C/LLVM 侧 packed function 的签名则来自 `MakePackedAPI` 后的 `PrimFunc` 参数：

```text
self_handle: handle
args: handle
num_args: int32
result: handle
return: int32
```

source C backend 会把它打印成 C 函数签名；LLVM backend 不走 `CodeGenC`，而是在
`CodeGenLLVM::DeclareFunction/AddFunction` 中生成对应 LLVM function type。

#### 5.2.3 AddFunction：打印函数定义和函数体

`CodeGenC::AddFunction(gvar, func)` 是函数体生成入口。

流程：

```text
AddFunction(gvar, func)
  -> DeclareFunction(gvar, func)
       如果前面已经声明过，这一步是 no-op
  -> function_name = GetFunctionName(gvar)
  -> InitFuncState(func)
       清空 var_idmap_、handle_data_type_、alloc_storage_scope_ 等
       ReserveKeywordsAsUnique()
  -> PrintFunctionSignature(function_name, func, stream)
  -> stream << " {"
  -> PreFunctionBody(func)
  -> func_scope = BeginScope()
  -> PrintStmt(func->body)
  -> EndScope(func_scope)
  -> stream << "}"
```

这里 `BeginScope/EndScope` 主要服务两个东西：

1. 控制缩进。
2. 管理 SSA 临时变量的生命周期，离开 scope 后临时表达式缓存失效。

`PreFunctionBody` 是子类插入函数体开头代码的钩子。本仓库 CUDA backend 里有一个例子：
如果函数带 `tirx.entry_cluster_sync`，`CodeGenCUDA::PreFunctionBody` 会先添加
`tvm_builtin_cuda_cluster_sync` helper，然后在 kernel 开头打印 cluster sync 调用。
本例没有这个 attr，所以 `PreFunctionBody` 不打印额外语句。

#### 5.2.4 PrintStmt：从函数体开始递归遍历

函数签名打印完后，真正的 IR 遍历从：

```cpp
this->PrintStmt(func->body);
```

开始。`PrintStmt` 只是 `VisitStmt(body)` 的薄封装，后面由 `StmtFunctor` 根据节点类型
分派到具体 overload：

```text
SeqStmt         -> 逐个 PrintStmt
AttrStmt       -> 处理 attr 后继续 PrintStmt(op->body)
For            -> 打印 for (...) { body }
IfThenElse     -> 打印 if (...) { ... }
BufferStore    -> PrintExpr(index/value)，再打印赋值
Evaluate(Call) -> 打印有副作用 call、sync、return 等
Bind           -> 打印局部变量定义，或在 SSA 模式登记 var 映射
AllocBuffer    -> 打印局部数组或 storage scope allocation
```

在语句中遇到表达式时，会调用 `PrintExpr(expr)`，进一步由 `ExprFunctor` 分派：

```text
Add/Sub/Mul    -> 二元表达式
Var            -> 查 var_idmap_
BufferLoad     -> 生成 buffer reference
Call           -> builtin / extern / internal GlobalVar / CUDA intrinsic
IntImm/FloatImm -> 常量
Cast           -> 类型转换
```

因此函数生成不是“把整个 PrimFunc 一次性模板化打印”，而是：

```text
函数签名先处理 params；
函数体从 Stmt root 深度优先递归；
语句负责结构，表达式负责值；
后端子类覆盖关键节点来改变目标语言语法。
```

#### 5.2.5 Finish：拼接源码片段

`CodeGenC` 生成源码时维护三段 stream：

| stream | 内容 |
| --- | --- |
| `decl_stream` | include/header、helper function、pragma import、CUDA header generator 输出。 |
| `fwd_decl_stream` | 第一轮 `DeclareFunction` 生成的 forward declarations。 |
| `stream` | 第二轮 `AddFunction` 生成的函数定义。 |

默认 `CodeGenC::Finish()` 返回：

```text
decl_stream + fwd_decl_stream + stream
```

CUDA 覆盖了 `Finish()`：它会先根据 codegen 过程中收集的 `codegen_tags_` 调
`tirx.intrinsics.cuda.header_generator(tags)` 生成 CUDA header，再把动态 intrinsic
helper source 插进 `decl_stream`，最后调用 `CodeGenC::Finish()` 拼接完整 CUDA source。

这也是为什么 CUDA source 开头会先出现一大段 include/type/helper，再出现：

```cuda
extern "C" __global__ void __launch_bounds__(2) main_kernel(...);
extern "C" __global__ void __launch_bounds__(2) main_kernel(...) {
  ...
}
```

### 5.3 表达式生成

常见表达式直接打印成 C 表达式：

| TIRX 节点 | C-style 输出 |
| --- | --- |
| `Var` | 已分配的变量名 |
| `IntImm` / `FloatImm` | 常量 |
| `Add/Sub/Mul/...` | 二元表达式 |
| `Cast` | C cast |
| `Select` | 三元表达式或临时变量 |
| `BufferLoad` | buffer reference |
| `Call(GlobalVar, args)` | 内部函数调用 |
| `Call(call_extern, ...)` | 外部函数调用 |

`BufferLoad/BufferStore` 要求 index 已 flatten 成一维；否则会报
`Load from non-flat memory not supported` 或 `Store to non-flat memory not supported`。

向量 load/store 有单独逻辑：如果 index 是连续 ramp 且满足对齐条件，会打印成
vector load/store；否则逐 lane 拆开。

### 5.4 语句生成

常见语句打印：

| TIRX 节点 | 输出 |
| --- | --- |
| `Bind` | 局部变量定义，或 SSA 映射。 |
| `AllocBuffer` | 局部数组或 storage-scope 数组。 |
| `AttrStmt(thread_extent)` | 绑定 thread index 变量，不直接打印 loop。 |
| `AttrStmt(pragma_import_c)` | 直接插入 C 源码到 declaration stream。 |
| `AssertStmt` | 生成错误处理或 `assert(...)`。 |
| `For` | C `for` loop。 |
| `While` | `while (1)` 加 break 条件。 |
| `IfThenElse` | C `if/else`。 |
| `Evaluate(Call(tvm_storage_sync))` | 后端自定义 sync 打印。 |
| `ExecScopeStmt` | 默认只打印 body。 |

`CodeGenC::VisitExpr_(CallNode)` 是 builtin lowering 的最后兜底。如果还有无法识别的
op，会直接报 `Unresolved call`，这通常意味着 `LowerIntrin` 或某个 target-specific
codegen registry 没有覆盖它。

## 6. Host C 后端

位置：`src/target/source/codegen_c_host.cc`

注册：

```cpp
refl::GlobalDef().def("target.build.c", BuildCHost);
```

`BuildCHost` 的流程：

```text
CodeGenCHost cg
  -> Init(output_ssa, emit_asserts, emit_fwd_func_decl, target_str, devices)
  -> 收集并排序 PrimFunc
  -> DeclareFunction
  -> AddFunction
  -> code = cg.Finish()
  -> CSourceModuleCreate(code, "c", function_names)
```

Host C 后端主要服务两类场景：

1. 没有 LLVM 时的 host fallback。
2. 需要查看或导出 C 源码时的 source module。

经过 `MakePackedAPI` 和 `LowerTVMBuiltin` 后，host 函数里常见的是 packed ABI、
DLTensor 参数检查、runtime call、workspace alloc/free、`tvm_call_packed` 等低层结构。

`CSourceModuleNode` 在 `src/target/source/source_module.cc`，它能：

| 方法 | 作用 |
| --- | --- |
| `InspectSource` | 返回源码字符串。 |
| `WriteToFile` | 写出 `.c/.cc/.cpp/.cu` 等文件。 |
| `SaveToBytes` | 序列化源码、format、function names。 |
| `GetFunction("get_func_names")` | 查询模块内函数名。 |

注意：source module 本身通常不能执行；执行需要对应 runtime/JIT 或导出后由外部编译。

## 7. CUDA 后端

CUDA 后端不是从零写一套 codegen。它继承 `CodeGenC`：

```cpp
class CodeGenCUDA final : public CodeGenC { ... };
```

因此它默认复用 C-style codegen 的大部分能力：

| 复用自 `CodeGenC` | CUDA 仍然直接使用的原因 |
| --- | --- |
| `DeclareFunction` / `AddFunction` 主流程 | CUDA C++ 也是 C-like source，可以复用函数声明、函数体打印和 visitor 递归。 |
| 大多数算术表达式打印 | `Add/Sub/Mul/Div/Compare/Select` 等在 CUDA C++ 中语法基本和 C 一致。 |
| `BufferLoad/BufferStore` 的 flat memory 访问框架 | CUDA 指针访问也是 `ptr[index]`，只需要补 storage/type/volatile 等细节。 |
| `For/If/While/SeqStmt/Bind` 等语句结构 | CUDA device code 仍然使用 C/C++ 控制流。 |
| `Call(GlobalVar)` / `call_extern` 基本机制 | device helper 调用和普通函数调用仍可按 C-style 打印。 |
| `decl_stream/fwd_decl_stream/stream` 拼接模型 | CUDA 也需要 header、forward declaration、函数定义三段源码。 |

CUDA 后端的核心工作，是在这套 C-style skeleton 上修正“C 不知道 GPU 语义”的地方：

```text
CodeGenC:
  会打印 C-like 函数、表达式、语句。

CodeGenCUDA:
  补 CUDA kernel/device 函数签名、
  补 thread/block/cluster 内建变量映射、
  补 __shared__/barrier/sync/storage 语义、
  补 CUDA 类型、vector、half/bf16/fp8/fp4、WMMA fragment，
  补 PTX/WMMA/TMA/cp.async 等 intrinsic，
  最后把源码包装成 CUDA runtime/fallback module。
```

位置：

| 文件 | 作用 |
| --- | --- |
| `src/target/cuda/codegen_cuda.h` | `CodeGenCUDA` 类声明。 |
| `src/target/cuda/codegen_cuda.cc` | CUDA source codegen 实现和 `target.build.cuda` 注册。 |
| `src/target/cuda/cuda_fallback_module.*` | CUDA runtime 不可用时保存源码的 fallback module。 |

注册：

```cpp
refl::GlobalDef().def("target.build.cuda", BuildCUDA);
```

### 7.1 BuildCUDA：沿用 C-style 函数生成，但限制输入是 device code

`BuildCUDA` 仍然走 `DeclareFunction -> AddFunction -> Finish`，和 `CodeGenC` 的流程一致。
它和通用 C 后端的差异在入口检查和 module 创建：

| 点 | C-style 基类/Host C | CUDA 修改 | 为什么 |
| --- | --- | --- | --- |
| 输入函数 | 普通 `tirx.PrimFunc` | 要求每个函数都是 `PrimFunc`，且 `calling_conv` 只能是 `kDeviceKernelLaunch` 或 `kDefault` | CUDA module 只应包含 device kernel 或 device helper，host packed entry 不应进入 CUDA codegen。 |
| codegen 类 | `CodeGenC` / `CodeGenCHost` | `CodeGenCUDA cg(target)` | 需要 target arch、CUDA 类型、intrinsic registry 等状态。 |
| 生成结果 | `CSourceModuleCreate(code, "c", ...)` | `CUDAModuleCreateWithFallback(source_bytes, "cuda", ExtractFuncInfo(mod), ...)` | CUDA source 需要被 CUDA runtime/JIT 或 fallback module 管理。 |
| 后处理 | Host C 无 CUDA postproc | 可调用 `tvm_callback_cuda_postproc(code, target)` | 允许外部 hook 改写 CUDA source。 |

流程：

```text
CodeGenCUDA cg(target)
  -> cg.Init(output_ssa=false)
  -> 检查每个函数都是 tirx.PrimFunc
  -> 检查 calling_conv 是 kDeviceKernelLaunch 或 kDefault
  -> DeclareFunction
  -> AddFunction
  -> code = cg.Finish()
  -> 可选 tvm_callback_cuda_postproc(code, target)
  -> CUDAModuleCreateWithFallback(source_bytes, "cuda", ExtractFuncInfo(mod), source_map)
```

`CUDAModuleCreateWithFallback` 的行为：

| 条件 | 结果 |
| --- | --- |
| TVM 编译时启用了 CUDA runtime 且未强制 fallback | 可通过 `tvm_callback_cuda_compile` JIT 编译成 PTX/cubin。 |
| 未启用 CUDA runtime 或强制 fallback | 返回保存 raw CUDA source 的 fallback module，供后续 cross compile。 |

### 7.2 函数签名：把 C 函数改成 CUDA kernel/device 函数

`CodeGenC` 默认打印的是普通 C 函数签名：

```c
void main_kernel(uint64_t* A_ptr)
```

CUDA 必须区分两类 device-side function：

`CodeGenCUDA::PrintFunctionSignature` 根据 calling convention 打印：

| `calling_conv` | CUDA 函数前缀 |
| --- | --- |
| `kDeviceKernelLaunch` | `extern "C" __global__` |
| `kDefault` | `extern "C" __device__` |

所以 `LowerDeviceKernelLaunch` 设置的 `calling_conv=kDeviceKernelLaunch` 会直接决定
这个函数最终是 CUDA kernel 还是 device helper。

为什么要改：

| CUDA 需求 | 如果沿用 C codegen 会怎样 |
| --- | --- |
| kernel 必须是 `__global__`，才能被 host runtime launch | 只是普通 C/device 函数，CUDA runtime 找不到可 launch kernel。 |
| device helper 必须是 `__device__`，才能从 kernel/device code 调用 | 普通 host 函数不能在 device code 中调用。 |
| kernel symbol 要稳定可见 | `extern "C"` 避免 C++ name mangling，runtime 按 `global_symbol` 查找 kernel。 |

`PrintExtraAttrs` 会从函数体中的 `thread_extent` 推导线程数，并打印
`__launch_bounds__(num_threads)`。persistent kernel 会打印
`__launch_bounds__(num_threads, 1)`。

这个修改直接作用到本例：

```text
calling_conv=2
threadIdx.x extent=2
tirx.noalias=True
```

最终签名：

```cuda
extern "C" __global__ void __launch_bounds__(2) main_kernel(
    uint64_t* __restrict__ A_ptr
);
```

### 7.3 thread binding：把 TIRX thread var 映射到 CUDA 内建变量

`CodeGenC::VisitStmt_(AttrStmtNode*)` 对 `thread_extent` 的默认处理是调用
`BindThreadIndex(iv)`，但基类不知道 `"threadIdx.x"` 应该怎么变成 CUDA builtin。
所以 CUDA 覆盖：

```cpp
void CodeGenCUDA::BindThreadIndex(const IterVar& iv)
```

`CodeGenCUDA::BindThreadIndex` 把 TIRX thread var 映射到 CUDA 内建变量：

```text
threadIdx.x / threadIdx.y / threadIdx.z
blockIdx.x / blockIdx.y / blockIdx.z
clusterCtaIdx.*
```

普通 thread tag 直接用同名 CUDA builtin；cluster CTA index 会生成小的 device helper，
从 special register 里读值。

为什么要改：

| TIRX IR | CUDA codegen 要做的事 |
| --- | --- |
| `AttrStmt(thread_extent, IterVar(threadIdx.x), 2)` | 不打印 loop，只把该 `Var` 绑定到 CUDA builtin `threadIdx.x`。 |
| `AttrStmt(thread_extent, IterVar(blockIdx.x), 6)` | 绑定到 `blockIdx.x`。 |
| `clusterCtaIdx.*` | CUDA C++ 没有普通 builtin，需要用 inline asm 读取 special register。 |

本例里：

```python
with T.launch_thread("blockIdx.x", 6) as blockIdx_x:
    threadIdx_x = T.launch_thread("threadIdx.x", 2)
    A[blockIdx_x * 2 + threadIdx_x] = ...
```

CUDA codegen 不会生成：

```cuda
for (...) { ... }
```

而是把两个变量替换成：

```cuda
blockIdx.x
threadIdx.x
```

所以最终 index 是：

```cuda
((((int)blockIdx.x) * 2) + ((int)threadIdx.x))
```

### 7.4 storage 和 sync：补齐 GPU memory hierarchy

`CodeGenC` 只知道普通 C 局部数组和指针。CUDA 需要表达不同 memory scope 和同步语义，
所以覆盖：

| 方法 | CUDA 修改 | 为什么 |
| --- | --- | --- |
| `PrintStorageScope` | 把 storage scope 打印成 CUDA qualifier，例如 `__shared__` | shared memory 必须在 CUDA source 里显式标注。 |
| `VisitStmt_(AllocBufferNode*)` | 特化 shared memory、dynamic shared memory、barrier array、WMMA fragment 等 allocation | GPU memory 的声明方式和普通 C stack array 不同。 |
| `PrintStorageSync` | 打印 `__syncthreads()`、barrier、warp/cluster sync 等 | `tvm_storage_sync` 不是 C 函数，必须变成 CUDA 同步原语。 |
| `PreFunctionBody` | 在需要时插入 cluster sync helper call | 某些 kernel 需要入口处同步 cluster。 |

为什么要改：如果沿用 C 的 `AllocBuffer`，shared memory 可能被打印成普通局部数组；
如果沿用 C 的 `tvm_storage_sync`，CUDA 编译器也不知道这是 `__syncthreads()`。

### 7.5 类型和向量：补 CUDA 类型系统

CUDA 子类覆盖了很多 C backend 行为：

| 方法 | CUDA 特化 |
| --- | --- |
| `PrintType` | 支持 half/bfloat16/fp8/fp4、CUDA vector type、WMMA fragment。 |
| `PrintVecConstructor` | 打印 CUDA vector constructor。 |
| `PrintVecElemLoad` / `PrintVecElemStore` | 处理 CUDA vector lane 访问。 |
| `PrintVecBinaryOp` | 处理 CUDA vector 二元操作。 |
| `CastFromTo` / `VisitExpr_(CastNode*)` | 处理 CUDA 特殊 dtype cast。 |
| `HandleVolatileLoads` | 处理 CUDA half volatile load 的兼容问题。 |

为什么要改：

| 类型问题 | C codegen 不够的地方 |
| --- | --- |
| `float16` | CUDA 用 `half` / `__half`，还需要 `<cuda_fp16.h>`。 |
| `bfloat16` | CUDA 有自己的 bf16 类型和 header。 |
| `fp8/fp6/fp4` | 需要 CUDA/NVIDIA 特定 storage 类型和 helper。 |
| vector dtype | CUDA 的 `int2/float4` 等构造和 lane 访问与普通 C 不完全一样。 |
| WMMA fragment | 不是普通数组类型，需要 `nvcuda::wmma::fragment<...>`。 |

`CodeGenCUDA` 会在发码过程中收集 `codegen_tags_`，例如 `fp16`、`bf16`、`fp8`、`mma`。
最后 `Finish()` 根据 tags 生成需要的 CUDA header/helper。

### 7.6 Call/intrinsic：把 CUDA op 变成 WMMA、PTX 或 helper call

`CodeGenC::VisitExpr_(CallNode*)` 只能处理通用 builtin、`call_extern`、内部
`GlobalVar` 调用等。CUDA 有大量 target-specific op，例如：

```text
WMMA
PTX mma/ldmatrix/cp.async
TMA
tcgen05
NVSHMEM
CUDA math/sync/memory intrinsic
```

所以 `CodeGenCUDA::VisitExpr_(CallNode*)` 在调用基类之前先做 CUDA 专用分发：

```text
VisitExpr_(CallNode)
  -> 如果 op 有 Python CUDA codegen registry:
       调 tirx.intrinsics.cuda.get_codegen(op.name)
       得到 cuda_func_call + tags
       打印 helper call，并记录 tags
  -> 否则匹配 C++ 中手写支持的 CUDA builtin:
       tvm_fill_fragment / tvm_load_matrix_sync / ptx_mma / ...
  -> 否则回退到 CodeGenC::VisitExpr_(CallNode)
```

TIRX CUDA intrinsic 有两套机制要分清。

第一套是 C++ `LowerIntrin` 的 op attr map：

```text
cuda.fastmath.FLowerIntrinsic
cuda.fastmath.FLegalize
cuda.FLowerIntrinsic
cuda.FLegalize
default.FLowerIntrinsic
default.FLegalize
```

位置：`src/tirx/transform/lower_intrin.cc`

它在 finalization pass 里运行，负责把 target-specific intrinsic 先改写成更基础的
表达式或 call。

第二套是 CUDA codegen 阶段的 Python registry：

| 入口 | 位置 |
| --- | --- |
| `tirx.intrinsics.cuda.get_codegen` | `python/tvm/tirx/operator/intrinsics/cuda/registry.py` |
| `register_codegen(op)` | 同上 |
| `device_intrinsic(...)` | `python/tvm/tirx/operator/intrinsics/_schema.py` |
| header generator | `python/tvm/tirx/operator/intrinsics/cuda/header.py` |

`CodeGenCUDA::VisitExpr_(CallNode)` 会：

```text
if op 是 Op:
  codegen = tirx.intrinsics.cuda.get_codegen(op.name)
  if codegen exists:
    func_call, tags = codegen(op.args)
    打印 func_call
    记录 tags
```

这些 Python codegen 通常返回一个 `cuda_func_call`，里面带：

| 内容 | 作用 |
| --- | --- |
| helper function name | 生成的 CUDA 调用名。 |
| forwarded args | 实际传给 helper 的 operands。 |
| source_code | 要插入到 CUDA 源码前面的 `__device__` helper。 |
| tags | header generator 需要的 include/helper tags。 |

`CodeGenCUDA::Finish` 最后调用：

```text
tirx.intrinsics.cuda.header_generator(tags)
```

生成必要 header，再把 util functions 插到 declaration stream。

例如 `python/tvm/tirx/operator/intrinsics/cuda/cp_async.py` 里，`device_intrinsic`
会生成含 inline PTX `asm volatile("cp.async...")` 的 CUDA helper，并注册到
`tirx.ptx_cp_async_*` op 的 codegen。

这个设计的好处是：新增很多 PTX form 时，不一定要改 C++ codegen；可以在 Python 侧
注册 op 和 helper codegen。

为什么要改：这些 op 不是标准 C 语法，也不是普通函数调用。比如 `cp.async` 最终要发
inline PTX，WMMA 要发 `nvcuda::wmma::*`，有些 form 还需要自动插入 helper function
和 header。基类 `CodeGenC` 无法知道这些 CUDA ISA 细节。

### 7.7 Finish 和 module 创建：CUDA source 不是普通 C source

`CodeGenC::Finish()` 只是拼：

```text
decl_stream + fwd_decl_stream + stream
```

CUDA 覆盖 `Finish()`，在拼接前多做两件事：

```text
CodeGenCUDA::Finish
  -> header_generator(codegen_tags_) 生成 CUDA headers/helpers
  -> 把 util_funcs_ 中的 helper source 插入 decl_stream
  -> CodeGenC::Finish()
```

然后 `BuildCUDA` 不返回普通 `CSourceModule`，而是：

```text
CUDAModuleCreateWithFallback(source_bytes, "cuda", ExtractFuncInfo(mod), source_map)
```

为什么要改：

| 普通 C source | CUDA source |
| --- | --- |
| 保存/导出成 C 文件即可 | 需要 CUDA runtime 或 NVRTC/NVCC 编译成 PTX/cubin。 |
| 函数通常由 CPU 直接调用 | kernel 由 runtime 通过 `global_symbol` launch。 |
| 无 import device module 语义 | CUDA module 会作为 host runtime module 的 imported module。 |

所以 CUDA 后端的整体心智模型是：

```text
CodeGenC 负责“像 C 一样递归打印 IR”；
CodeGenCUDA 负责“把其中和 GPU/CUDA 相关的节点改成合法 CUDA C++/PTX”；
BuildCUDA 负责“把 CUDA source 包装成可 JIT/可 fallback 的 runtime module”。
```

## 8. LLVM 后端

位置：

| 文件 | 作用 |
| --- | --- |
| `src/target/llvm/llvm_module.cc` | `target.build.llvm` 注册和 `LLVMModuleNode`。 |
| `src/target/llvm/codegen_llvm.h/.cc` | 通用 LLVM IR codegen。 |
| `src/target/llvm/codegen_cpu.h/.cc` | CPU host 特化，runtime call、parallel、packed call 等。 |
| `src/target/llvm/codegen_x86_64.cc` / `codegen_arm.cc` / `codegen_aarch64.cc` | 架构特化。 |

注册：

```cpp
refl::GlobalDef().def("target.build.llvm", [](IRModule mod, Target target) {
  auto n = ffi::make_object<LLVMModuleNode>();
  n->Init(mod, target);
  return ffi::Module(n);
});
```

`LLVMModuleNode::Init(mod, target)` 的主线：

```text
LLVMTarget llvm_target(target)
CodeGenLLVM::Create(llvm_target)
  -> 根据 target triple/cpu 选择 CodeGenCPU 或架构子类
cg->Init("TVMMod", ...)
cg->SetFastMathFlags(...)
cg->AddFunctionsOrdered(mod->functions.begin(), mod->functions.end())
if entry_func:
  cg->AddMainFunction(entry_func)
module = cg->Finish()
设置 target metadata / debug metadata / system lib metadata
```

`CodeGenLLVM` 和 `CodeGenC` 类似，也是直接继承 TIRX visitor：

```cpp
ExprFunctor<llvm::Value*(const PrimExpr&)>
StmtFunctor<void(const Stmt&)>
```

但它不是打印字符串，而是用 `llvm::IRBuilder` 构造 LLVM IR：

| TIRX 节点 | LLVM codegen 方向 |
| --- | --- |
| `Var` | 查 `var_map_` 中的 `llvm::Value*`。 |
| `Add/Sub/Mul/...` | 创建 LLVM arithmetic instruction。 |
| `BufferLoad` | 计算 pointer，生成 load。 |
| `BufferStore` | 计算 pointer，生成 store。 |
| `For` | 生成 basic blocks 和 PHI/branch。 |
| `IfThenElse` | 生成 then/else/end basic blocks。 |
| `Call` | builtin/runtime/intrinsic/internal function call。 |

`CodeGenCPU` 在通用 LLVM codegen 上增加 host runtime 语义，例如：

| 能力 | 代码点 |
| --- | --- |
| packed call | `CodeGenCPU::MakeCallPackedLowered` / `CreateCallPacked` |
| runtime function lookup | `RuntimeTVMFFIFunctionCall`、`RuntimeTVMGetFuncFromEnv` |
| parallel launch | `CreateParallelLaunch` |
| assert 错误返回 | `VisitStmt_(AssertStmtNode*)` |
| entry main function | `AddMainFunction` |

所以 host target 是 `llvm` 时，packed ABI 和 runtime 调用最终会变成真正可 JIT/导出
的 LLVM module。

## 9. Trainium 后端

位置：`src/target/source/codegen_trn.cc`

注册：

```cpp
refl::GlobalDef().def("target.build.trn", BuildTrainium);
```

Trainium 是一个比较特殊的 source backend。它不是生成 C/CUDA，而是生成 NKI Python：

```python
import neuronxcc.nki.language as nl
from neuronxcc.nki import baremetal, benchmark, simulate_kernel, trace
...
@baremetal(...)
def kernel(...):
    ...
```

`BuildTrainium` 的主线：

```text
for each PrimFunc:
  CodeGenTrainium cg(target)
  cg.Init(output_ssa)
  cg.AddFunction(gvar, prim_func)
  fsource = cg.Finish()
  smap[func_name] = fsource

return DeviceSourceModuleCreate(source_maker, fmt, ExtractFuncInfo(mod), "nki")
```

`fmt` 取决于是否注册了 `tvm_callback_Trainium_compile`：

| 条件 | fmt |
| --- | --- |
| 有 `tvm_callback_Trainium_compile` | `Trainiumlib` |
| 没有 | `Trainium` |

Trainium 的 TIRX pipeline 也不同：

```text
TrnPrivateBufferAlloc
TrnNaiveAllocator
LowerTIRx
DecorateDeviceScope
...
finalize_device_passes_trn = Simplify only
```

这说明 target codegen 的契约可以按后端调整：CUDA/LLVM 更依赖标准 finalization；
TRN 在 pipeline 中提前做自己的 allocation 和 instruction generation。

## 10. Runtime Module 的形态

codegen 的产物统一是 `ffi::Module` / `runtime.Module`，但不同 target 的 module 类型不同：

| 后端 | 产物形态 |
| --- | --- |
| `llvm` | `LLVMModuleNode`，可 JIT、导出 object/shared library。 |
| `c` | `CSourceModuleNode`，保存 C source。 |
| `cuda` | CUDA runtime module 或 fallback source module。 |
| `opencl/metal/webgpu` | 对应 source/runtime module。 |
| `vulkan` | SPIR-V / Vulkan module。 |
| `trn` | NKI device source module。 |

host module 通过 `import_module` 挂 device modules。序列化/导出时，
`src/target/codegen.cc` 里的 `ModuleSerializer` 会处理 import tree，并把可 DSO 导出的
module 合并到合适的位置。

## 11. 从 IR 节点到后端代码的对应关系

下面是一张学习时很好用的“对照表”。

| IR/attr | 何时出现 | codegen 处理 |
| --- | --- | --- |
| `PrimFunc.params` | 函数 ABI 参数 | 打印函数参数或 LLVM function args。 |
| `PrimFunc.buffer_map` | packed API 前的结构化 buffer | `MakePackedAPI` 消费，codegen 阶段更多看 handle/pointer 参数。 |
| `global_symbol` | 导出符号 | 决定最终函数名。 |
| `calling_conv=kCPackedFunc` | host entry | host backend 生成 packed function 入口。 |
| `calling_conv=kDeviceKernelLaunch` | device kernel | CUDA 打印 `__global__`，device runtime 导出 kernel。 |
| `calling_conv=kDefault` | internal/helper func | CUDA 打印 `__device__`，LLVM/C 打印普通函数。 |
| `thread_extent` | thread binding | CUDA 绑定 thread var；LowerDeviceKernelLaunch 收集 launch args。 |
| `AllocBuffer` | 局部 buffer | C/CUDA 打印局部数组或 storage scope allocation。 |
| `BufferLoad/Store` | 内存读写 | 后端生成 load/store；要求 flatten。 |
| `Call(GlobalVar)` | 内部函数调用或 launch 前形式 | same-target 保留为内部调用；cross-target 先由 pass 改写。 |
| `call_extern` | 外部 C/device helper 调用 | C-style backend 直接打印函数调用。 |
| `tvm_call_packed` | runtime packed call | host finalization/codegen 降成 TVM FFI runtime 调用。 |
| target intrinsic op | CUDA/PTX/WMMA 等 | `LowerIntrin` 或 CUDA codegen registry 处理。 |

## 12. 调试和阅读路线

### 12.1 看 build 边界

优先读：

```text
python/tvm/tirx/build.py
python/tvm/tirx/compilation_pipeline.py
src/tirx/transform/annotate_device_regions.cc
src/tirx/transform/split_host_device.cc
src/tirx/transform/lower_device_kernel_launch.cc
```

目标是弄清楚：某个 `PrimFunc` 什么时候变成 host function，什么时候变成 device
kernel，launch 参数在哪里被收集。

### 12.2 看 source backend

然后读：

```text
src/target/source/codegen_source_base.h
src/target/source/codegen_c.h
src/target/source/codegen_c.cc
src/target/cuda/codegen_cuda.cc
```

建议按 visitor 读，不要从头到尾顺序读：

1. `AddFunction`
2. `PrintFunctionSignature`
3. `VisitStmt_(AttrStmtNode*)`
4. `VisitStmt_(ForNode*)`
5. `VisitExpr_(BufferLoadNode*)`
6. `VisitStmt_(BufferStoreNode*)`
7. `VisitExpr_(CallNode*)`
8. CUDA 子类覆盖的同名方法

### 12.3 看 LLVM backend

CPU/LLVM 路径读：

```text
src/target/llvm/llvm_module.cc
src/target/llvm/codegen_llvm.h
src/target/llvm/codegen_llvm.cc
src/target/llvm/codegen_cpu.h
src/target/llvm/codegen_cpu.cc
```

重点是 `LLVMModuleNode::Init` 到 `CodeGenLLVM::AddFunctionsOrdered` 的链路，
再对照 `CodeGenLLVM::VisitExpr_` / `VisitStmt_`。

### 12.4 看生成源码

source backend 生成的 module 通常可以 inspect：

```python
rt_mod = tvm.tirx.build(mod, target="cuda", pipeline="tirx")
print(rt_mod.imports[0].inspect_source("cuda"))
```

如果 host 是 C source，也可以：

```python
print(rt_mod.inspect_source("c"))
```

具体 format 取决于返回的 module kind，例如 `cuda`、`c`、`nki`。

### 12.5 看 pass 后 IR

想定位 codegen 报错前 IR 是否满足契约，可以单独跑 pipeline 或在 pass 中间打印。
已有一个 TIRX dispatch 后打印开关：

```text
TVM_PRINT_AFTER_TIRX_DISPATCH_OPS=1
```

也可以直接在 Python 里跑：

```python
pipeline, finalize_host, finalize_device = tvm.tirx.get_tir_pipeline("tirx")
lowered = pipeline(mod)
print(lowered.script())
```

如果 `CodeGenC` 报 unresolved call，通常按这个顺序查：

1. 对应 op 是否应该被 `LowerIntrin` 改写。
2. CUDA registry 是否注册了 `tirx.intrinsics.cuda.get_codegen(op.name)`。
3. `Call` 是否仍是高层 tile primitive 产物，说明 `LowerTIRx` 没跑或 dispatch 失败。
4. 是否传错 pipeline，例如需要 `pipeline="tirx"` 却走了默认 `s_tir`。

## 13. 从一个 UT 例子走完整 codegen

这里选一个简单但有代表性的 UT：

```text
3rdparty/tvm/tests/python/codegen/test_target_codegen_device.py::test_large_uint_imm
```

它只有一个 12 元素 GPU kernel，但覆盖了完整 codegen 主链路：

```text
tvm.compile
  -> tvm.tirx.build
  -> BindTarget
  -> default/s_tir pipeline
  -> AnnotateDeviceRegions
  -> SplitHostDevice
  -> MakePackedAPI
  -> LowerDeviceKernelLaunch
  -> split_host_device_mods
  -> target.build.cuda / target.build.llvm
  -> host module import CUDA module
```

UT 源码核心如下：

```python
@tvm.testing.requires_gpu
def test_large_uint_imm():
    value = (1 << 63) + 123
    value_const = tvm.tirx.const(value, "uint64")

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer((12,), "uint64")):
            T.func_attr({"tirx.noalias": True})
            for i0_0 in T.thread_binding(6, thread="blockIdx.x"):
                for i0_1 in T.thread_binding(2, thread="threadIdx.x"):
                    with T.sblock("A"):
                        v_i0 = T.axis.spatial(12, i0_0 * 2 + i0_1)
                        T.reads()
                        T.writes(A[v_i0])
                        A[v_i0] = value_const + T.uint64(3)

    f = tvm.compile(Module, target="cuda")
    a = tvm.runtime.empty((12,), dtype="uint64", device=tvm.cuda(0))
    f(a)
    assert a.numpy()[0] == value + 3
```

### 13.1 这个例子为什么合适

它足够小：

| 结构 | 说明 |
| --- | --- |
| 一个 `main` PrimFunc | 入口简单。 |
| 一个 buffer 参数 `A` | packed API 参数处理容易看。 |
| 两层 `thread_binding` | 会产生 `blockIdx.x` 和 `threadIdx.x` launch 参数。 |
| 一个 `sblock` | 会经过 S-TIR/TIRX pipeline 清理。 |
| 一个 `BufferStore` | 最终能直接对应 CUDA store。 |
| 一个 `large_uint_imm` | 顺手覆盖大整数常量 codegen。 |

它也足够代表真实 GPU 编译：

1. 输入 module 最初只有一个函数。
2. build 后会出现 host wrapper 和 device kernel 两个函数。
3. host wrapper 负责 packed ABI、参数检查、设备设置和 kernel launch。
4. CUDA module 负责真正执行 `A[index] = const + 3`。

### 13.2 Step 0：前端 IR 形状

TVMScript parser 会把 `value_const` 打印成 `T.large_uint_imm`：

```python
A[v_i0] = T.large_uint_imm(T.uint32(123), T.uint32(2147483648)) + T.uint64(3)
```

原因是 `value = 2^63 + 123` 超过 signed int64 的正数范围，TIRX 用两个
`uint32` 保存低/高 32 位：

```text
low  = 123
high = 2147483648
value = (high << 32) | low
```

此时 IR 里仍然是单个 `main`：

```text
PrimFunc main(A)
  attrs: tirx.noalias=True, s_tir=True
  for blockIdx.x in thread_binding(6)
    for threadIdx.x in thread_binding(2)
      SBlock("A")
        A[blockIdx.x * 2 + threadIdx.x] = large_uint_imm(...) + 3
```

注意：这里虽然写了 `thread_binding`，但还没有真正生成 CUDA kernel。它只是
IR 中的 thread extent 信息，后续 pass 会决定这段代码属于 device region。

### 13.3 Step 1：tvm.compile 转到 tirx.build

UT 调用：

```python
f = tvm.compile(Module, target="cuda")
```

非 Relax module 会走：

```text
tvm.compile
  -> tvm.tirx.build(Module, target="cuda", pipeline="default")
```

`tirx.build` 做 target 决策：

| 项 | 结果 |
| --- | --- |
| `target_to_bind` | `cuda` |
| `target` | `cuda` |
| `target_host` | 通常是 `llvm`，如果 LLVM 不可用才是 `c` |
| 绑定 target | `cuda.with_host(llvm)` |

之后先跑：

```text
tirx.transform.BindTarget(cuda.with_host(llvm))
```

这一步让原始 `main` 带上 target 信息。因为 target 有 host，后续 pipeline 才能把
host wrapper 和 CUDA kernel 分出来。

### 13.4 Step 2：default pipeline 生成 host + device 函数

这个 module 标记了 `s_tir=True`，`pipeline="default"` 当前会映射到 `"s_tir"`。
关键 pass 的效果如下：

| pass | 在本例中的效果 |
| --- | --- |
| `LowerInitBlock` / `UnifyThreadBinding` / `Simplify` | 清理 S-TIR block 和 thread binding 形态。 |
| `FlattenBuffer` | 保证 `A[...]` 是一维 flat access。 |
| `AnnotateEntryFunc` | 标记入口函数。 |
| `AnnotateDeviceRegions` | 看到 `thread_extent`，把这段 body 标成 CUDA device region。 |
| `SplitHostDevice` | 抽出 `main_kernel` device function，host `main` 中留下调用。 |
| `MakePackedAPI` | 把 host `main(A)` 改成 TVM packed ABI 入口。 |
| `LowerDeviceKernelLaunch` | 把 host 对 `main_kernel` 的调用改成 `T.call_packed("main_kernel", A_ptr, 6, 2)`，并给 kernel 加 launch attrs。 |

pipeline 后的 module 用 `lowered.script(show_meta=True)` 打印出来如下。target attrs
里的 `arch`、`mtriple` 会随机器环境变化；其余结构就是这个例子的 lowered 形态。

```python
metadata = tvm.ir.load_json("""{
  "root_index": 17,
  "nodes": [
    {
      "type": "ffi.String",
      "data": "tirx.StringImm"
    },
    {
      "type": "None"
    },
    {
      "type": "ffi.String",
      "data": " when calling:\\n  `"
    },
    {
      "type": "tirx.StringImm",
      "data": {
        "span": 1,
        "dtype": "handle",
        "value": 2
      }
    },
    {
      "type": "ffi.String",
      "data": "`,\\n  expected "
    },
    {
      "type": "tirx.StringImm",
      "data": {
        "span": 1,
        "dtype": "handle",
        "value": 4
      }
    },
    {
      "type": "tirx.StringImm",
      "data": {
        "span": 1,
        "dtype": "handle",
        "value": 4
      }
    },
    {
      "type": "tirx.StringImm",
      "data": {
        "span": 1,
        "dtype": "handle",
        "value": 4
      }
    },
    {
      "type": "tirx.StringImm",
      "data": {
        "span": 1,
        "dtype": "handle",
        "value": 4
      }
    },
    {
      "type": "tirx.StringImm",
      "data": {
        "span": 1,
        "dtype": "handle",
        "value": 4
      }
    },
    {
      "type": "ffi.String",
      "data": "`,\\n  expected to be compact array"
    },
    {
      "type": "tirx.StringImm",
      "data": {
        "span": 1,
        "dtype": "handle",
        "value": 10
      }
    },
    {
      "type": "ffi.String",
      "data": "`,\\n  expected non-NULL data pointer"
    },
    {
      "type": "tirx.StringImm",
      "data": {
        "span": 1,
        "dtype": "handle",
        "value": 12
      }
    },
    {
      "type": "tirx.StringImm",
      "data": {
        "span": 1,
        "dtype": "handle",
        "value": 4
      }
    },
    {
      "type": "tirx.StringImm",
      "data": {
        "span": 1,
        "dtype": "handle",
        "value": 4
      }
    },
    {
      "type": "ffi.Array",
      "data": [3, 5, 6, 7, 8, 9, 11, 13, 14, 15]
    },
    {
      "type": "ffi.Map",
      "data": [0, 16]
    }
  ],
  "metadata": {
    "tvm_version": "0.25.dev0"
  }
}""")

@I.ir_module
class Module:
    @T.prim_func(s_tir=True)
    def main_kernel(A_ptr: T.handle("uint64", "global")):
        T.func_attr({
            "calling_conv": 2,
            "target": T.target({
                "arch": "sm_89",
                "keys": ["cuda", "gpu"],
                "kind": "cuda",
                "max_num_threads": 1024,
                "tag": "",
                "thread_warp_size": 32,
            }),
            "tirx.is_global_func": True,
            "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x"],
            "tirx.noalias": True,
        })
        A = T.decl_buffer((12,), "uint64", data=A_ptr)
        with T.launch_thread("blockIdx.x", 6) as blockIdx_x:
            threadIdx_x = T.launch_thread("threadIdx.x", 2)
            A[blockIdx_x * 2 + threadIdx_x] = (
                T.large_uint_imm(T.uint32(123), T.uint32(2147483648)) + T.uint64(3)
            )

    @T.prim_func(s_tir=True)
    def main(self_handle: T.handle, args: T.handle, num_args: T.int32,
             result: T.handle("void", "global")) -> T.int32:
        T.func_attr({
            "calling_conv": 1,
            "global_symbol": "__tvm_ffi_main",
            "target": T.target({
                "keys": ["cpu"],
                "kind": "llvm",
                "mtriple": "x86_64-pc-linux-gnu",
                "tag": "",
            }),
            "tirx.is_entry_func": True,
            "tirx.noalias": True,
        })
        assert num_args == 1, (
            "TypeError",
            ["Expected ", "1", " arguments", metadata["tirx.StringImm"][0],
             "main(A: Tensor([12], uint64))", "`"],
        )
        assert not T.isnullptr(args), (
            "TypeError",
            ["args pointer is NULL", metadata["tirx.StringImm"][0],
             "main(A: Tensor([12], uint64))", "`"],
        )
        A_handle_type_index: T.let[T.int32] = T.tvm_struct_get(args, 0, 13, "int32")
        assert (
            A_handle_type_index == 0
            or A_handle_type_index == 4
            or A_handle_type_index == 7
            or A_handle_type_index >= 64
        ), (
            "TypeError",
            ["Mismatched type on argument #", "0", metadata["tirx.StringImm"][0],
             "main(A: Tensor([12], uint64))", metadata["tirx.StringImm"][1], "Tensor"],
        )
        A_handle: T.let[T.handle] = T.Select(
            A_handle_type_index == 70,
            T.handle_add_byte_offset(T.tvm_struct_get(args, 0, 15, "handle"), 24),
            T.tvm_struct_get(args, 0, 15, "handle"),
        )
        assert not T.isnullptr(A_handle), (
            "TypeError",
            ["Mismatched type on argument #", "0", metadata["tirx.StringImm"][0],
             "main(A: Tensor([12], uint64))", metadata["tirx.StringImm"][2], "Tensor"],
        )
        assert 1 == T.tvm_struct_get(A_handle, 0, 4, "int32"), (
            "ValueError",
            ["Mismatched ", "A", ".ndim on argument #", "0",
             metadata["tirx.StringImm"][0], "main(A: Tensor([12], uint64))",
             metadata["tirx.StringImm"][3], "1"],
        )
        assert (
            T.tvm_struct_get(A_handle, 0, 5, "uint8") == T.uint8(1)
            and T.tvm_struct_get(A_handle, 0, 6, "uint8") == T.uint8(64)
            and T.tvm_struct_get(A_handle, 0, 7, "uint16") == T.uint16(1)
        ), (
            "TypeError",
            ["Mismatched ", "A", ".dtype on argument #", "0",
             metadata["tirx.StringImm"][0], "main(A: Tensor([12], uint64))",
             metadata["tirx.StringImm"][4], "uint64"],
        )
        main_A_handle_shape: T.let[T.handle] = T.tvm_struct_get(A_handle, 0, 2, "handle")
        main_A_handle_strides: T.let[T.handle] = T.tvm_struct_get(A_handle, 0, 3, "handle")
        assert T.tvm_struct_get(A_handle, 0, 10, "int32") == 2, (
            "ValueError",
            ["Mismatched ", "A", ".device_type on argument #", "0",
             metadata["tirx.StringImm"][0], "main(A: Tensor([12], uint64))",
             metadata["tirx.StringImm"][5], "cuda"],
        )
        dev_id: T.let[T.int32] = T.tvm_struct_get(A_handle, 0, 9, "int32")
        A_ptr: T.let[T.handle("uint64", "global")] = T.tvm_struct_get(
            A_handle, 0, 1, "handle"
        )
        with T.attr(A_ptr, "storage_alignment", 64):
            T.attr("default", "device_id", dev_id)
            T.attr("default", "device_type", 2)
            if not T.isnullptr(main_A_handle_strides):
                assert 1 == T.Cast(
                    "int32", T.tvm_struct_get(main_A_handle_strides, 0, 17, "int64")
                ), (
                    "ValueError",
                    ["Mismatched ", "A", ".strides on argument #", "0",
                     metadata["tirx.StringImm"][0], "main(A: Tensor([12], uint64))",
                     metadata["tirx.StringImm"][6]],
                )
            assert not T.isnullptr(A_ptr), (
                "ValueError",
                ["A", " data pointer is NULL on argument #", "0",
                 metadata["tirx.StringImm"][0], "main(A: Tensor([12], uint64))",
                 metadata["tirx.StringImm"][7]],
            )
            assert T.Cast(
                "int32", T.tvm_struct_get(main_A_handle_shape, 0, 17, "int64")
            ) == 12, (
                "ValueError",
                ["Invalid ", "A.shape[0]", " on argument #", "0",
                 metadata["tirx.StringImm"][0], "main(A: Tensor([12], uint64))",
                 metadata["tirx.StringImm"][8], "12"],
            )
            assert T.uint64(0) == T.tvm_struct_get(A_handle, 0, 8, "uint64"), (
                "ValueError",
                ["Invalid ", "A.byte_offset", " on argument #", "0",
                 metadata["tirx.StringImm"][0], "main(A: Tensor([12], uint64))",
                 metadata["tirx.StringImm"][9], "0"],
            )
            A = T.decl_buffer((12,), "uint64", data=A_ptr)
            T.call_packed("__tvm_set_device", 2, dev_id)
            with T.attr({"compute_scope": "main_compute_"}):
                A_1 = T.decl_buffer((12,), "uint64", data=A_ptr)
                T.call_packed("main_kernel", A_ptr, 6, 2)
            return 0
```

两个 attrs 最关键：

| attr | 含义 |
| --- | --- |
| `calling_conv=1` | host entry 是 packed function。 |
| `calling_conv=2` | device function 是 kernel launch function。 |

`tirx.kernel_launch_params = ["blockIdx.x", "threadIdx.x"]` 表示 host launch 时要额外
传两个 runtime launch 参数。对应值来自 thread extent：

```text
blockIdx.x  -> 6
threadIdx.x -> 2
```

所以 host 里出现：

```python
T.call_packed("main_kernel", A_ptr, 6, 2)
```

### 13.5 Step 3：split_host_device_mods 再按 target 分 module

pipeline 结束时，`main` 和 `main_kernel` 还在同一个 `IRModule`。
`build.py::split_host_device_mods` 再分成：

```text
host_mod:
  main        target=llvm

device_mod_dict[cuda]:
  main_kernel target=cuda
```

分组规则很直接：

```text
target.kind.name in ["llvm", "c"] -> host
otherwise                         -> device
```

这一步之后，host 和 device 会分别跑 finalization pass。

### 13.6 Step 4：finalization pass

本例的 host finalization：

```text
LowerTVMBuiltin
LowerCustomDatatypes
LowerIntrin
```

作用是把 host wrapper 里的 TVM builtin、packed call、datatype/intrinsic 处理到
LLVM/C backend 能接受的形态。

本例的 device finalization：

```text
LowerWarpMemory
Simplify
LowerCustomDatatypes
LowerIntrin
```

这个 kernel 很简单，没有 warp memory 和复杂 intrinsic；主要会做最后的清理和
常量/intrinsic 合法化。

### 13.7 Step 5：device codegen 先生成 CUDA source

`tir_to_runtime` 会先编译 device module：

```text
codegen_build(cuda_device_mod, cuda)
  -> target.build.cuda
  -> BuildCUDA
  -> CodeGenCUDA
```

`BuildCUDA` 检查 `main_kernel` 的 `calling_conv=2`，于是
`CodeGenCUDA::PrintFunctionSignature` 打印：

```cuda
extern "C" __global__
```

`CodeGenCUDA::PrintExtraAttrs` 从 `threadIdx.x` extent 推导线程数是 `2`，于是打印：

```cuda
__launch_bounds__(2)
```

最终 CUDA source 的关键部分是：

```cuda
extern "C" __global__ void __launch_bounds__(2) main_kernel(
    uint64_t* __restrict__ A_ptr
);

extern "C" __global__ void __launch_bounds__(2) main_kernel(
    uint64_t* __restrict__ A_ptr
) {
  A_ptr[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] =
      ((uint64_t)9223372036854775931 + (uint64_t)3);
}
```

这里能直接对上前面的 IR：

| IR | CUDA source |
| --- | --- |
| `tirx.noalias=True` | `__restrict__ A_ptr` |
| `calling_conv=2` | `__global__` |
| `threadIdx.x extent=2` | `__launch_bounds__(2)` |
| `blockIdx.x` / `threadIdx.x` | CUDA builtin variables |
| `large_uint_imm(123, 2147483648)` | `9223372036854775931` |
| `BufferStore A[...]` | `A_ptr[...] = ...` |

然后 CUDA source 被交给：

```text
CUDAModuleCreateWithFallback
```

如果当前 TVM 启用了 CUDA runtime，就可以 JIT 成可执行 CUDA module；否则保存源码作为
fallback module。

### 13.8 Step 6：host codegen 生成 packed wrapper

device module 编完后，host module 再走：

```text
codegen_build(host_mod, llvm)
  -> target.build.llvm
  -> LLVMModuleNode::Init
  -> CodeGenLLVM / CodeGenCPU
```

host wrapper 做的不是数值计算，而是运行时工作：

1. 检查 `num_args == 1`。
2. 从 TVM packed args 里取出 `A` 的 DLTensor handle。
3. 检查 `A.ndim == 1`、`A.dtype == uint64`、`A.shape[0] == 12`。
4. 检查 `A.device_type == cuda`。
5. 取出 `A_ptr`。
6. 调 `__tvm_set_device`。
7. 调 `main_kernel`，并传入 launch args `6, 2`。
8. 返回 `0`。

这解释了为什么用户侧只需要：

```python
f(a)
```

host wrapper 会把 Python/runtime 层的 `NDArray` 解包成 device pointer，并通过
runtime launch CUDA kernel。

### 13.9 Step 7：runtime module 组装

最后：

```text
host_runtime = target.build.llvm(host_mod)
dev_runtime  = target.build.cuda(device_mod)
host_runtime.import_module(dev_runtime)
return host_runtime
```

所以返回的 `f` 是 host module，但它的 import tree 里挂了 CUDA module。这个关系很
重要：host wrapper 里只有 `call_packed("main_kernel", A_ptr, 6, 2)`，真正实现
`main_kernel` 的函数不在 host module 自己里面，而在 imported CUDA module 里面。

调用时的查找顺序是：

```text
f(a)
  -> 进入 host packed entry main
  -> host code 调 runtime packed call: "main_kernel"
  -> host module 查自己是否有 main_kernel
  -> 查不到，再查 imports[0] 的 CUDA module
  -> CUDA module 返回 CUDAWrappedFunc
  -> CUDAWrappedFunc 发射 main_kernel<<<6, 2>>>(A_ptr)
```

因此 `import_module` 是 host/device codegen 的最后一个连接点。前面的 pass 负责把
host 调 device 表示成 packed call；CUDA codegen 负责把 device kernel 做成可被
`GetFunction("main_kernel")` 找到的 runtime function；`import_module` 负责让 host
module 的运行时查找能看到这个 device function。

调试时可以看 import tree 里的 CUDA source：

```python
rt_mod = tvm.tirx.build(Module, target="cuda")
print(rt_mod.imports[0].inspect_source("cuda"))
```

UT 最后执行：

```python
a = tvm.runtime.empty((12,), dtype="uint64", device=dev)
f(a)
assert a.numpy()[0] == value + 3
```

运行路径是：

```text
Python 调 f(a)
  -> LLVM host packed wrapper
  -> unpack/check A
  -> tvm_call_packed("main_kernel", A_ptr, 6, 2)
  -> CUDA runtime launch main_kernel<<<6, 2>>>(A_ptr)
  -> 每个 thread 写一个 A[index]
  -> 拷回 CPU 检查 A[0]
```

这个例子的 index 空间正好是：

```text
6 blocks * 2 threads = 12 elements
index = blockIdx.x * 2 + threadIdx.x
```

### 13.10 从这个例子得到的 codegen 读法

读任何 GPU codegen 问题时，可以照这个顺序定位：

1. 原始 IR 是否有 `thread_binding` / `launch_thread`。
2. pipeline 后是否出现 device `*_kernel`。
3. kernel 是否有 `calling_conv=2` 和 `tirx.kernel_launch_params`。
4. host wrapper 里是否出现 `T.call_packed("kernel", ..., launch_args...)`。
5. `split_host_device_mods` 后 host/device 是否按 target 分组。
6. device `inspect_source("cuda")` 里函数是否是 `__global__`。
7. 参数、thread index、buffer store 是否和 IR 对得上。

这比直接盯着 `CodeGenCUDA` 看更稳，因为很多“codegen 问题”实际是前面某个 pass
没有把 IR 降到 codegen 期望的契约形态。

## 14. 新增 CUDA intrinsic 的实现路径

如果要给 CUDA 增加一个新 intrinsic，大致有两种方式。

### 14.1 Python helper registry 路径

适合发一段 helper function 或 inline PTX。

1. 在 `python/tvm/tirx/operator/intrinsics/cuda/*.py` 注册 codegen：

```python
from .._schema import device_intrinsic

device_intrinsic(
    "my_intrin",
    c_signature="(int x)",
    body='    asm volatile("...");',
)
```

2. 用户侧 wrapper 产生 `call_intrin("", "tirx.my_intrin", ...)` 或对应 op call。
3. `CodeGenCUDA::VisitExpr_(CallNode)` 在 codegen 阶段查 registry。
4. registry 返回 `cuda_func_call`，CodeGenCUDA 插入 helper source 并打印调用。
5. 如果需要 include 或辅助代码，在返回 tags 里加 header tag。

这个路径不需要改 C++ `CodeGenCUDA`，适合 PTX form 多、变化快的 intrinsic。

### 14.2 C++ LowerIntrin 路径

适合把某个 op legalize 成更基础的 IR 或 target builtin。

1. 给 op 注册 `cuda.FLowerIntrinsic` / `cuda.FLegalize` attr。
2. `tirx.transform.LowerIntrin` 在 finalization 中匹配 op。
3. 改写后的 IR 再进入 `CodeGenCUDA` 或 `CodeGenC` 默认 call 逻辑。

这个路径适合语义层面的 lowering，而不是单纯发一段源代码 helper。

## 15. 新增 target backend 的最小路径

如果要新增一个 target kind，比如 `foo`：

1. 定义 target kind，使 `Target("foo")` 可用。
2. 确定 pipeline：复用 `tirx`/`s_tir`，或者在 `PIPELINE_MAP` 注册自定义 pipeline。
3. 实现 `target.build.foo`：

```cpp
ffi::Module BuildFoo(IRModule mod, Target target) {
  // 检查 mod 里都是 tirx.PrimFunc
  // 遍历函数，生成目标代码
  // 包装成 runtime/source module
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.foo", BuildFoo);
}
```

4. 如果是 C-like 语言，优先继承 `CodeGenC`，覆盖：

```text
PrintFunctionSignature
PrintType
PrintStorageScope
PrintStorageSync
VisitExpr_(CallNode*)
VisitStmt_(AllocBufferNode*)
```

5. 如果是 binary/IR target，参考 LLVM/Vulkan，直接生成对应中间表示。
6. 返回一个 `ffi::Module`，至少支持 `InspectSource` 或执行/导出能力之一。

## 16. 一句话心智模型

TIRX codegen 的核心不是“把 Python DSL 翻译成 CUDA”，而是：

```text
前端构造 TIRX IR；
pass 把高层 TIRX 语义降成 target-ready PrimFunc；
build.py 按 target 拆分、finalize、分发；
target.build.<kind> 用 TIRX visitor 生成目标代码；
runtime.Module import tree 把 host 和 device code 组装成一个可执行/可导出的模块。
```

读源码时抓住三条线：

1. `build.py`：谁调用谁、什么时候分 host/device、什么时候调用 `target.build.*`。
2. `split_host_device.cc` + `lower_device_kernel_launch.cc`：kernel 函数和 launch ABI 怎么形成。
3. `CodeGenC` / `CodeGenCUDA` / `CodeGenLLVM`：每类 TIRX 节点最终怎么被打印或生成。
