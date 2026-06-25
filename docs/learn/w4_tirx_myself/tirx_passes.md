# TIRX IR Pass 梳理

本文基于当前仓库 `3rdparty/tvm` 的源码，梳理 `tvm.tirx` 这套 IR 的
transform pass。重点看这些入口：

| 入口 | 作用 |
| --- | --- |
| `3rdparty/tvm/include/tvm/tirx/transform.h` | C++ 侧公开的 TIRX transform pass 声明。 |
| `3rdparty/tvm/src/tirx/transform/*.cc` | pass 的主要实现。 |
| `3rdparty/tvm/src/tirx/analysis/verify_*.cc` | 注册在 `tirx.transform` 命名空间下的校验 pass。 |
| `3rdparty/tvm/python/tvm/tirx/transform/transform.py` | Python 侧 wrapper。 |
| `3rdparty/tvm/python/tvm/tirx/compilation_pipeline.py` | `tvm.tirx.build` 使用的预置 lowering pipeline。 |
| `3rdparty/tvm/python/tvm/s_tir/pipeline.py` | `s_tir` pipeline，里面混用了一部分 `tirx.transform.*` pass。 |

需要先区分两件事：

- `tirx.transform.*` 是 TIRX 自己的 pass；它处理 `tirx.PrimFunc`、
  `tirx.Stmt`、`tirx.PrimExpr`、`tirx.Buffer`、`ExecScopeStmt`、
  `TilePrimitiveCall` 等节点。
- `s_tir.transform.*` 是 TensorIR/S-TIR 侧 pass。TIRX 的默认构建流程会混用
  S-TIR pass，但本文只展开 TIRX 自己的 pass；S-TIR pass 只在 pipeline 位置
  点名。

## 1. Pass 基础设施

| API | 来源 | 作用 |
| --- | --- | --- |
| `CreatePrimFuncPass` | `src/tirx/ir/transform.cc` | 创建只作用于 `tirx.PrimFunc` 的 pass。它会遍历 `IRModule` 中的 `tirx.PrimFunc`，把 C++/Python 的函数级变换包装成 TVM pass。 |
| `prim_func_pass` | `python/tvm/tirx/transform/function_pass.py` | Python 自定义 `PrimFuncPass` 装饰器，类似 TVM 常规 `function_pass`，但输入输出类型是 `tirx.PrimFunc`。 |
| `Apply` | `python/tvm/tirx/transform/transform.py` | 轻量 wrapper：把一个 `PrimFunc -> PrimFunc` 的 Python 函数包装成 TIRX pass。 |
| `Filter` | `src/tirx/transform/primfunc_utils.cc` | 按用户给的 predicate 保留或删除 `PrimFunc`。predicate 返回 false 时该函数会被过滤掉。 |

## 2. TIRX lowering 主线 pass

这组 pass 负责把 TIRX 高层 IR 降到更接近传统低层 TIR/codegen 的形态。

| Pass | 类型 | 作用 |
| --- | --- | --- |
| `LowerTIRx` | `Sequential` | TIRX 专用 lowering 总入口。当前组合为 `TilePrimitiveDispatch -> LowerTIRxCleanup -> LowerTIRxStripExecScope`，中间可通过环境变量 `TVM_PRINT_AFTER_TIRX_DISPATCH_OPS` 打印 IR。 |
| `TilePrimitiveDispatch` | `PrimFuncPass` | 降低 `TilePrimitiveCall`。它根据 target、执行 scope、launch 参数、buffer workspace 等上下文，调用已注册的 tile primitive dispatcher，把 `Tx.copy/Tx.gemm/Tx.reduce` 等高层 tile op 改写成低层语句或 intrinsic 调用。还会解析 `ScopeIdDef`，生成 thread/block launch 参数，并校验 lowering 后不再残留 `TilePrimitiveCall`。 |
| `LowerTIRxCleanup` | `PrimFuncPass` | `TilePrimitiveDispatch` 后的清理。主要移除 dispatch context，应用 layout flatten/rewrite，清掉 buffer offset 等 TIRX lowering 中间信息。 |
| `LowerTIRxOpaque` | `PrimFuncPass` | 降低 TIRX opaque 构造，类似 `s_tir.LowerOpaqueBlock` 的 TIRX 版本，但不处理 `SBlock`。它处理 `AllocBuffer`、`For(thread_binding)` 到 `AttrStmt(thread_extent)`、unit loop 消除、pragma annotation 保留/转换等。 |
| `LowerTIRxStripExecScope` | 内部 `PrimFuncPass` | `LowerTIRx` 内部 pass，去掉 `ExecScopeStmt` wrapper。`ExecScopeStmt` 在 tile op dispatch 和 scope id 解析时有用，最终低层 IR 不应继续携带它。 |

## 3. 常规 IR 简化与规范化

| Pass | 类型 | 作用 |
| --- | --- | --- |
| `Simplify` | `PrimFuncPass` | 对语句和表达式做算术化简，底层使用 `arith::StmtSimplifier`。常在 buffer flatten、unroll、TIRX lowering 后重复跑，用于消掉冗余表达式和简化条件。 |
| `RemoveNoOp` | `PrimFuncPass` | 删除 no-op 语句，例如无意义的 `Evaluate(0)`、空 `SeqStmt`、恒假/恒真分支中的无效结构等。可用 `tirx.RemoveNoOp` 配置启用 dataflow 分析、限制 rewrite 步数、忽略 profiler call。 |
| `RemoveAssume` | `Sequential` | 删除 `tirx::builtin::assume()` 调用，然后跑 `RemoveNoOp` 清理产生的空语句。内部实际有 `RemoveAssumeInternal`。 |
| `SkipAssert` | `PrimFuncPass` | 直接移除 `AssertStmt`，替换为 `Evaluate(0)`。通常只适合明确不需要运行时断言的场景。 |
| `ConvertSSA` | `ModulePass` | 把模块里的 TIRX IR 转成 SSA 形态，解决同一个 `tirx.Var` 在多个函数或多个作用域里重复定义的问题。`SplitHostDevice`、`InlinePrivateFunctions` 后常需要它。 |
| `InlinePrivateFunctions` | `ModulePass` | 内联 private `PrimFunc` 调用。它会收集可内联函数，把 `Evaluate(Call(GlobalVar, ...))` 形式的调用替换成 callee body 的 specialization，并删除不再需要的 private callee，最后跑 `ConvertSSA` 修复变量定义。 |
| `CommonSubexprElim` | `PrimFuncPass` | 公共子表达式消除。先规划可复用表达式，再用 `Bind` 引入临时变量，减少重复计算。 |

## 4. Loop、vector、datatype 与 buffer pass

| Pass | 类型 | 作用 |
| --- | --- | --- |
| `FlattenBuffer` | `PrimFuncPass` | 把多维 `BufferLoad/BufferStore` flatten 成一维访问，并更新外部 `buffer_map`。它要求 `MatchBufferRegion` 等更高层 buffer 结构已经被降低。 |
| `VectorizeLoop(enable_vectorize=True)` | `PrimFuncPass` | 降低 `ForKind::kVectorized` loop。开启时把 vectorized loop 改写成向量表达式/向量 load-store；关闭时把 vectorized loop 退化成 serial loop。 |
| `UnrollLoop` | `PrimFuncPass` | 展开标记为 unroll 的常量循环，也会根据配置自动给满足条件的 loop 做 unroll。配置项是 `tirx.UnrollLoop`，包括 `auto_max_step`、`auto_max_depth`、`auto_max_extent`、`explicit_unroll`、`unroll_local_access` 等。 |
| `NarrowDataType(target_bits)` | `PrimFuncPass` | 把表达式里的整数 dtype 尽量收窄到目标 bit 数，默认 pipeline 常用 `NarrowDataType(32)`，通常放在 `FlattenBuffer` 后。 |
| `ForceNarrowIndexToInt32` | `PrimFuncPass` | 强制把 index 表达式和整数 buffer 收窄到 `int32`。源码注释明确说默认场景不应使用，属于更激进的工具 pass。 |
| `StorageRewrite` | `PrimFuncPass` | 重写 storage allocation：把 allocation 尽量移动到外层作用域，规划静态内存复用，减少临时 buffer 占用。之后还会对内部 allocation 做 `PointerValueTypeRewrite`。受 `tirx.merge_static_smem` 影响，开启 shared memory 合并时会关闭本 pass 的复用逻辑。 |
| `PointerValueTypeRewrite` | `PrimFuncPass` | 根据实际 load/store 的最常用向量类型，重写 pointer 参数、内部 `Alloc`/`AllocBuffer` 的元素类型，减少 backend 中的 pointer cast。 |

## 5. Host/device 拆分与 ABI pass

| Pass | 类型 | 作用 |
| --- | --- | --- |
| `BindTarget(target)` | `ModulePass` | 给模块里的 `PrimFunc` 绑定 host/device target，并根据调用关系处理 host/device 函数副本和调用替换。后续 target 相关 pass 依赖这个属性。 |
| `AnnotateEntryFunc` | `ModulePass` | 推断并标记入口函数。如果模块只有一个函数，或只有一个带 `global_symbol` 的外部 `PrimFunc`，就给它加 `tirx.is_entry_func`。 |
| `AnnotateDeviceRegions` | `PrimFuncPass` | 在带 host 的 target 下，把包含 `thread_extent` 或 `device_scope` 的区域包上 `AttrStmt(target=target.WithoutHost())`，标出哪些区域应该作为 device 代码。 |
| `SplitHostDevice` | `ModulePass` | 根据 `AttrStmt(target=...)` 把 host 函数里的 device region 抽成独立 device `PrimFunc`。新 device 函数会带 device target、`tirx.noalias`、`tirx.is_global_func` 等属性，host 侧留下对 device 函数的调用。最后会跑 `ConvertSSA`。 |
| `MakePackedAPI` | `ModulePass` | 把外部可调用的 `PrimFunc` 改写成 TVM packed ABI。它会消费 `buffer_map`，生成参数 unpack、DLTensor shape/stride/offset 检查、动态 shape var 绑定和返回码，并把函数参数改成 packed API 形式。 |
| `LowerDeviceKernelLaunch` | `ModulePass` | 把 host 到 device 的普通函数调用降成 `tvm_call_packed`，并把 device 函数更新成外部可见的 kernel symbol，同时收集和传递 launch 参数。 |

## 6. Builtin、intrinsic、target lowering pass

| Pass | 类型 | 作用 |
| --- | --- | --- |
| `LowerTVMBuiltin` | `PrimFuncPass` | 只作用于 host 函数，把 TVM builtin 调用降成更底层的 runtime/FFI 调用序列，例如 packed call、内存分配、设备信息处理等。 |
| `LowerCustomDatatypes` | `PrimFuncPass` | 降低自定义 datatype。它依赖 TVM datatype registry，把自定义类型运算改写成 target/codegen 可理解的普通调用或表达式。 |
| `LowerIntrin` | `PrimFuncPass` | 降低 target-specific intrinsic。它要求函数有 `target` 属性，根据 target 的 intrinsic lowering rule 把 `tirx.call_intrin` 等改写成后端能发码的形式；受 `tirx.enable_fast_math` 影响。 |
| `LowerWarpMemory` | `PrimFuncPass` | 降低 warp scope memory 访问。根据 target 的 `thread_warp_size` 重写 warp memory 访问，并更新 pointer storage scope，通常在 device finalization 阶段使用。 |
| `RemapThreadAxis(axis_map)` | `PrimFuncPass` | 把一种 thread axis 映射成另一种，例如把 `threadIdx.x` 改成 `threadIdx.y`，同时替换 thread extent attr、相关变量引用和 kernel launch params。C++/FFI 有注册，但当前 Python `transform.py` 没有显式 wrapper。 |

## 7. BF16/FP8 legalize pass

这组 pass 用来处理 target 不原生支持的低精度类型。源码会先检查 CUDA target
是否支持对应类型，支持则跳过，不支持才改写。

| Pass | 类型 | 作用 |
| --- | --- | --- |
| `BF16ComputeLegalize` | `PrimFuncPass` | 对 bf16 计算做合法化：计算前提升到 fp32，计算后再 cast 回 bf16。 |
| `FP8ComputeLegalize(promote_dtype)` | `PrimFuncPass` | 对 fp8 计算做合法化：计算前提升到 fp16/fp32，计算后再 cast 回 fp8。C++ 默认参数是 `float16`，当前 Python wrapper 默认是 `float32`。 |
| `BF16StorageLegalize` | `PrimFuncPass` | 把 bf16 storage 改写成等宽无符号整数，一般是 `uint16`，避免不支持 bf16 storage 的后端直接处理 bf16 指针/存储。 |
| `FP8StorageLegalize` | `PrimFuncPass` | 把 fp8 storage 改写成等宽无符号整数，一般是 `uint8`。源码注释要求它依赖 target 属性，通常应在 target 绑定之后跑。 |

## 8. 校验 pass

这两个实现位于 `src/tirx/analysis`，但通过 FFI 注册到了 `tirx.transform.*`。

| Pass | 类型 | 作用 |
| --- | --- | --- |
| `VerifySSA` | `ModulePass` | 遍历模块里的 `tirx.PrimFunc`，检查变量是否满足 SSA：每个 `Var` 定义一次，使用点在定义作用域内。不满足时抛 `RuntimeError`。Python analysis API 也有 `tirx.analysis.verify_ssa(func)`。 |
| `VerifyMemory` | `ModulePass` | 检查是否存在非法 host 侧直接访问 device memory 的情况。典型错误是 CUDA kernel 没有正确 bind thread，导致生成 host 代码直接读写 GPU buffer。不通过时抛 `RuntimeError` 并提示 “Did you forget to bind?”。 |

## 9. Trainium 专用 Python pass

`python/tvm/tirx/transform/trn` 下有两个 lazy import 的 TIRX pass，主要服务
Trainium pipeline。

| Pass | 类型 | 作用 |
| --- | --- | --- |
| `trn.TrnPrivateBufferAlloc` | Python `PrimFuncPass` | 针对 `TilePrimitiveCall` 收集 private workspace 需求，在外层 `ExecScopeStmt` 注入 private buffer allocation 和初始化语句，并把 workspace 回填到 tile primitive call。 |
| `trn.TrnNaiveAllocator` | Python `PrimFuncPass` | 给 `trn.sbuf` 或 shared scope 的 `AllocBuffer` 分配线性地址。已有 `allocated_addr` 的 buffer 会保留，未分配的 buffer 按大小顺序追加。要求 shape 为常量。 |

## 10. TIRX 编译流程与代码位置

TIRX 的编译入口主要是 `tvm.tirx.build`。它不是单个 C++ pass，而是一条
Python 侧 orchestration 流程：先把输入变成 `IRModule`，绑定 target，选择并
执行 lowering pipeline，再拆 host/device module，分别做 finalization，最后调用
各 target 注册的 `target.build.<kind>` 生成 runtime module。

### 10.1 入口位置

| 入口 | 代码位置 | 说明 |
| --- | --- | --- |
| `tvm.tirx.build` | `python/tvm/tirx/build.py::build` | TIRX 编译主入口。 |
| `tirx.build` global func | `python/tvm/tirx/build.py` 文件末尾 | `tvm.register_global_func("tirx.build", build)`，把 Python build 注册成全局函数。 |
| `from tvm import tirx; tirx.build` | `python/tvm/tirx/__init__.py` | 非 runtime-only 模式下导入 `.build import build`，同时导入 compilation pipeline。 |
| `tvm.build(...)` | `python/tvm/driver/build_module.py::build` | 兼容入口，当前会 warning 并转调 `tvm.tirx.build`。 |
| `tvm.compile(...)` | `python/tvm/driver/build_module.py::compile` | 如果输入不含 Relax function，会调用 `tvm.tirx.build(..., pipeline=tir_pipeline)`，再包成 `runtime.Executable`。 |
| Relax VM build | `python/tvm/relax/vm_build.py` | Relax build 在 lowering 出 TIR module 后，也会调用 `tvm.tirx.build(tir_mod, target=target, pipeline=tir_pipeline)`。 |

### 10.2 主流程

`python/tvm/tirx/build.py::build` 的实际步骤如下：

```text
PrimFunc / IRModule
  1. PrimFunc -> IRModule.from_expr
  2. 决定 target_to_bind
  3. 决定用于选择 pipeline 的 target
  4. 决定 host target: llvm 或 c，或 target.host
  5. target_to_bind.with_host(target_host)
  6. BindTarget(target_to_bind)
  7. 选择并执行 tir pipeline
  8. split_host_device_mods
  9. host/device finalization passes
 10. codegen_build(device_mod, device_target)
 11. codegen_build(host_mod, target_host)
 12. host runtime module import device runtime module
```

逐步解释如下：

| # | 流程 | 代码位置 | 简单说明 |
| --- | --- | --- | --- |
| 1 | `PrimFunc -> IRModule.from_expr` | `build.py::build` Step 0 前 | `build` 同时接受单个 `tirx.PrimFunc` 和 `IRModule`。如果传入的是 `PrimFunc`，先包成只含一个函数的 `IRModule`，这样后续 pass 都按模块处理。 |
| 2 | 决定 `target_to_bind` | `build.py::build` Step 0 | 这是后面 `BindTarget` 使用的 target。优先取用户传入的 `target`，否则取 `Target.current()`，还没有就退到 `"llvm"`。 |
| 3 | 决定用于选择 pipeline 的 `target` | `build.py::build` Step 1 | 这是选择默认 pipeline 时看的 target。优先取用户传入的 `target`，否则尝试从已有 `PrimFunc.attrs["target"]` 里找一个。 |
| 4 | 决定 `target_host` | `build.py::build` Step 2 | host 端默认用 `llvm`，如果 LLVM 没开则用 `c`。如果 device target 自带 host，例如 `cuda -host=llvm`，就用 target 里指定的 host。 |
| 5 | `target_to_bind.with_host(target_host)` | `build.py::build` Step 2 | 把 host target 塞回待绑定 target，形成带 host/device 信息的 target。后续 host/device split 和 packed API 需要知道 host 侧应该怎么编。 |
| 6 | `BindTarget(target_to_bind)` | `build.py::build` Step 3，`src/tirx/transform/bind_target.cc` | 给模块里的 `PrimFunc` 补齐或传播 target 属性。没有 target 的函数会获得默认 target，跨 host/device 调用也会在这里整理。 |
| 7 | 选择并执行 TIR pipeline | `build.py::build` Step 4，`compilation_pipeline.py` | 如果 `pipeline` 是字符串，调用 `get_tir_pipeline(pipeline)`；默认 `"default"` 会映射到 `"s_tir"`。拿到 pipeline 后执行 `mod = pipeline(mod)`。 |
| 8 | `split_host_device_mods` | `build.py::split_host_device_mods` | pipeline 里的 `SplitHostDevice` 已经把 host/device 函数拆到同一个 `IRModule` 中；这里再按函数 target 把它们分成一个 host module 和若干 device module。 |
| 9 | host/device finalization passes | `build.py::build` Step 6，`compilation_pipeline.py` | host module 跑 `finalize_host_passes()`，通常是 `LowerTVMBuiltin/LowerCustomDatatypes/LowerIntrin`；device module 跑 `finalize_device_passes()`，通常会处理 warp memory、intrinsic 等。 |
| 10 | `codegen_build(device_mod, device_target)` | `build.py::tir_to_runtime`，`build.py::codegen_build` | 先编译每个非空 device module。`codegen_build` 会查 `target.build.<device_kind>`，例如 CUDA 走 `target.build.cuda`。 |
| 11 | `codegen_build(host_mod, target_host)` | `build.py::tir_to_runtime`，`build.py::codegen_build` | 再编译 host module。host 一般走 `target.build.llvm` 或 `target.build.c`，产物是主 runtime module。 |
| 12 | host module import device modules | `build.py::tir_to_runtime` | 最后把每个 device runtime module 通过 `mhost.import_module(dev_mod)` 挂到 host runtime module 下，返回一个包含 host 和 device code 的 `tvm.runtime.Module`。 |

### 10.3 Pipeline 选择与注册

`python/tvm/tirx/compilation_pipeline.py` 维护 `PIPELINE_MAP`：

```text
PIPELINE_MAP = {
  "default": default_tir_pipeline,
  "tirx": tirx_pipeline,
  "trn": trn_pipeline,
}
```

但这里有两个跨包注册点：

| 注册点 | 代码位置 | 作用 |
| --- | --- | --- |
| `PIPELINE_MAP["s_tir"] = default_s_tir_pipeline` | `python/tvm/s_tir/pipeline.py` | `get_tir_pipeline("default")` 会先把 `"default"` 改成 `"s_tir"`，所以默认路径实际依赖 S-TIR pipeline 被导入注册。 |
| `PIPELINE_MAP["adreno"] = default_tir_pipeline` | `python/tvm/s_tir/backend/adreno/pipeline.py` | Adreno target 的默认 pipeline 来自 S-TIR Adreno backend。 |

因此常见路径是：

```text
tvm.tirx.build(mod, target, pipeline="default")
  -> get_tir_pipeline("default")
  -> "default" 映射到 "s_tir"
  -> 使用 tvm.s_tir.pipeline.default_s_tir_pipeline
```

如果明确传 `pipeline="tirx"`，才会走 `compilation_pipeline.py::tirx_pipeline()`。
如果传 `pipeline="trn"`，会走 Trainium 专用 pipeline。

### 10.4 Codegen 后端位置

`codegen_build` 不直接知道 CUDA/LLVM 怎么发码，它只按 target kind 查全局函数：

```python
build_f_name = "target.build." + target.kind.name
bf = tvm.get_global_func(build_f_name)
return bf(mod, target)
```

常见后端注册位置：

| Target kind | 全局函数 | 代码位置 |
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

Python 的通用 target codegen wrapper 在 `python/tvm/target/codegen.py::build_module`，
底层对应 C++ `src/target/codegen.cc::Build`；而 TIRX build 当前直接在
`build.py::codegen_build` 中查 `target.build.<kind>`。

### 10.5 编译流程和 pass 的关系

可以把前面各 pass 放回编译流程里看：

```text
TVMScript / Python API
  -> tirx.PrimFunc / IRModule

tvm.tirx.build
  -> BindTarget
  -> selected pipeline
       LowerTIRx / LowerTIRxOpaque
       Simplify / FlattenBuffer / dtype legalize / vectorize / unroll / CSE
       VerifyMemory
       AnnotateEntryFunc
       AnnotateDeviceRegions
       SplitHostDevice
       MakePackedAPI
       FP8StorageLegalize / BF16StorageLegalize
       LowerDeviceKernelLaunch
  -> split_host_device_mods
  -> host finalization
       LowerTVMBuiltin
       LowerCustomDatatypes
       LowerIntrin
  -> device finalization
       LowerWarpMemory
       Simplify
       LowerCustomDatatypes
       LowerIntrin
  -> target.build.<device>
  -> target.build.<host>
  -> host module import device modules
  -> tvm.runtime.Module
```

关键边界是：`SplitHostDevice` 是 IRModule 内部的 host/device 函数拆分；
`split_host_device_mods` 是 build 阶段把已经带不同 target 的函数分进不同
`IRModule`，用于分别调用 target codegen。

### 10.6 `pipeline="default"` 涉及的全部 pass

`tvm.tirx.build` 的函数签名默认是 `pipeline="default"`。在当前代码里，
`get_tir_pipeline("default")` 会把 `"default"` 改成 `"s_tir"`，所以实际跑的是
`python/tvm/s_tir/pipeline.py::default_s_tir_pipeline()`。

注意这里说的是显式或默认参数 `pipeline="default"`。如果传 `pipeline=None`，
代码会走 `get_default_tir_pipeline(target)`，那是另一条“按 target 选择默认
pipeline”的路径。

#### 10.6.1 Build 外层 pass

这些 pass 不在 `default_s_tir_pipeline()` 内部，但属于
`tvm.tirx.build(..., pipeline="default")` 的编译流程。

| 顺序 | Pass | 来源 | 作用 |
| --- | --- | --- | --- |
| 1 | `tirx.transform.BindTarget(target_to_bind)` | `python/tvm/tirx/build.py` Step 3；实现：`src/tirx/transform/bind_target.cc` | 在进入 lowering pipeline 前给函数绑定 target，补齐 host/device target 属性。 |

#### 10.6.2 `default_s_tir_pipeline` 主体 pass

代码位置：`python/tvm/s_tir/pipeline.py::default_s_tir_pipeline()`。

这 55 个 pass 可以先按阶段分成 8 类：

| 阶段 | 顺序范围 | Pass | 主要目的 |
| --- | --- | --- | --- |
| SBlock/loop 规范化 | 1-6 | `CanonicalizeLoop`, `LowerCrossThreadReduction`, `LowerInitBlock`, `PlanAndUpdateBufferAllocationLocation`, `ConvertBlocksToOpaque`, `LiftThreadBinding` | 把还带调度语义的 block/loop 结构变得更规范，为后续 lowering 做准备。 |
| Buffer 与 block 降低 | 7-18 | `ManifestSharedMemoryLocalStage`, `CompactBufferAllocation`, `LowerAutoCopy`, `UnifyThreadBinding`, `LowerMatchBuffer`, `Simplify`, `InjectPermutedLayout`, `AnnotateIrregularLoop`, `InjectSoftwarePipeline`, `TransformMmaBufferLayout`, `LowerOpaqueBlock`, `FlattenBuffer` | 处理 shared/local buffer、match buffer、software pipeline、MMA layout，并最终把 block/多维 buffer 访问降到更低层形式。 |
| 类型、循环和内存优化 | 19-32 | `BF16ComputeLegalize`, `NarrowDataType`, `LoopPartition`, `VectorizeLoop`, `InjectVirtualThread`, `InjectDoubleBuffer`, `StorageRewrite`, `LowerAsyncDMA`, `HoistIfThenElse`, `UnrollLoop`, `RenormalizeSplitPattern`, `Simplify`, `RemoveNoOp`, `RewriteUnsafeSelect` | 做 dtype 合法化、loop partition/vectorize/unroll、storage rewrite、async DMA lowering 和 IR 清理。 |
| 可选调试/性能插桩 | 33-36 | `InstrumentBoundCheckers`, `InjectPTXLDG32(True)`, `CommonSubexprElim`, `InstrumentProfileIntrinsics` | 根据 `PassContext` 配置插入检查、PTX ldg32、CSE 或 profiling intrinsic。 |
| 低精度/VTCM/内存校验 | 37-41 | `FP8ComputeLegalize`, `VerifyVTCMLimit`, `LowerVtcmAlloc`, `VerifyMemory`, `AnnotateEntryFunc` | 处理 fp8 compute、VTCM allocation/限制、内存合法性，并标注入口函数。 |
| 线程同步与 GPU collective lowering | 42-48 | `ThreadSync("shared")`, `ThreadSync("shared.dyn")`, `ThreadSync("warp")`, `InferFragment`, `LowerThreadAllreduce`, `InjectPTXAsyncCopy`, `InjectPTXLDG32()` | 插入 shared/warp 同步，推断 tensorcore fragment，降低 thread allreduce 和 CUDA async copy/ldg32。 |
| Host/device 拆分与 ABI | 49-52 | `AnnotateDeviceRegions`, `SplitHostDevice`, `MergeSharedMemoryAllocations`, `MakePackedAPI` | 标注 device 区域，抽出 device kernel，合并 shared memory allocation，并把外部入口改成 packed API。 |
| Storage legalize 与 kernel launch lowering | 53-55 | `FP8StorageLegalize`, `BF16StorageLegalize`, `LowerDeviceKernelLaunch` | 把低精度 storage 改成后端可处理的整数存储，并把 host 到 device kernel 的调用改成 runtime launch。 |

下面是严格按照源码顺序展开的完整表：

| 顺序 | Pass | 命名空间 | 作用 |
| --- | --- | --- | --- |
| 1 | `CanonicalizeLoop` | `s_tir.transform` | 把 loop 规范化成从 0 开始，方便后续分析和改写。 |
| 2 | `LowerCrossThreadReduction` | `s_tir.transform` | 把跨线程 reduction 从 thread binding 形式降低成 intrinsic 调用。 |
| 3 | `LowerInitBlock` | `s_tir.transform` | 把 block init 语句降低成 `IfThenElse` 等普通控制流。 |
| 4 | `PlanAndUpdateBufferAllocationLocation` | `s_tir.transform` | 规划 buffer allocation 的位置，通常放到访问点的 LCA，并在 allocation site 注入 opaque block。 |
| 5 | `ConvertBlocksToOpaque` | `s_tir.transform` | 把 block var 替换成对应 `iter_values`，把可调度 block 转成 opaque block。 |
| 6 | `LiftThreadBinding` | `s_tir.transform` | 把相同 thread binding 提升到公共外层 loop。 |
| 7 | `ManifestSharedMemoryLocalStage` | `s_tir.transform` | 为 GPU shared memory 访问显式加入 local stage。 |
| 8 | `CompactBufferAllocation` | `s_tir.transform` | 根据实际访问区域收窄临时 buffer shape，减少分配空间。 |
| 9 | `LowerAutoCopy` | `s_tir.transform` | 对 auto copy block 做内存相关优化和降低。 |
| 10 | `UnifyThreadBinding` | `s_tir.transform` | 统一同一 thread axis 的 `IterVar` 和变量，比如多个 `threadIdx.x` 使用同一个绑定。 |
| 11 | `LowerMatchBuffer` | `s_tir.transform` | 移除 block 内部的 match buffer，并校验绑定合法性。 |
| 12 | `Simplify` | `tirx.transform` | 算术和语句化简。 |
| 13 | `InjectPermutedLayout` | `s_tir.transform` | 为 shared memory 注入 permuted layout。 |
| 14 | `AnnotateIrregularLoop` | `s_tir.transform` | 标注 irregular loop，供后续 pass/codegen 使用。 |
| 15 | `InjectSoftwarePipeline` | `s_tir.transform` | 根据 loop annotation 生成 software pipeline 的 prologue/body/epilogue。 |
| 16 | `TransformMmaBufferLayout` | `s_tir.transform` | 把 MMA scope buffer 转换到 local scope，并做 layout transformation。 |
| 17 | `LowerOpaqueBlock` | `s_tir.transform` | 移除 opaque block，让 IR 不再可 schedule。 |
| 18 | `FlattenBuffer` | `tirx.transform` | 把多维 buffer 访问 flatten 成一维访问。 |
| 19 | `BF16ComputeLegalize` | `tirx.transform` | 对不支持 bf16 原生计算的 target，把 bf16 compute 合法化。 |
| 20 | `NarrowDataType(32)` | `tirx.transform` | 把可收窄的整型表达式降到 32 bit。 |
| 21 | `LoopPartition` | `s_tir.transform` | 对 loop 做 partition，常用于拆出边界条件或简化循环体。 |
| 22 | `VectorizeLoop` | `tirx.transform` | 降低 vectorized loop；受 `tirx.disable_vectorize` 控制，关闭时退化为 serial loop。 |
| 23 | `InjectVirtualThread` | `s_tir.transform` | 注入 virtual thread loop。 |
| 24 | `InjectDoubleBuffer` | `s_tir.transform` | 注入 double buffer 相关语句。 |
| 25 | `StorageRewrite` | `tirx.transform` | 重写 storage allocation，做静态内存规划和复用；当 `tirx.disable_storage_rewrite=True` 时跳过。 |
| 26 | `LowerAsyncDMA` | `s_tir.transform` | 把 async TIR primitive 降成 DMA copy/wait builtin；仅 `tirx.use_async_copy=True` 时插入。 |
| 27 | `HoistIfThenElse` | `s_tir.transform` | 把 loop-invariant 的条件语句提升到合适的外层 loop。 |
| 28 | `UnrollLoop` | `tirx.transform` | 展开显式或自动判定可展开的常量 loop。 |
| 29 | `RenormalizeSplitPattern` | `s_tir.transform` | 规范化 split 后的 `floordiv/floormod` 模式。 |
| 30 | `Simplify` | `tirx.transform` | 第二次化简，清理 unroll/hoist 后产生的表达式。 |
| 31 | `RemoveNoOp` | `tirx.transform` | 删除 no-op 语句和空结构。 |
| 32 | `RewriteUnsafeSelect` | `s_tir.transform` | 改写带内存访问的 unsafe select，避免 select 两侧访问都被求值造成问题。 |
| 33 | `InstrumentBoundCheckers` | `s_tir.transform` | 插入边界检查；仅 `tirx.instrument_bound_checkers=True` 时插入。 |
| 34 | `InjectPTXLDG32(True)` | `s_tir.transform` | CUDA 上把 global-to-local copy 改写为 ldg32 指令形式；仅 `tirx.ptx_ldg32=True` 时插入。 |
| 35 | `CommonSubexprElim` | `tirx.transform` | 公共子表达式消除；当 `tirx.disable_cse_tir=True` 时跳过。 |
| 36 | `InstrumentProfileIntrinsics` | `s_tir.transform` | 插入函数/loop 级 profiling intrinsic；仅 `tirx.instrument_lwp=True` 时插入。 |
| 37 | `FP8ComputeLegalize` | `tirx.transform` | 对不支持 fp8 原生计算的 target，把 fp8 compute 合法化。 |
| 38 | `VerifyVTCMLimit` | `s_tir.transform` | 校验 VTCM 使用量是否超过 target 限制。 |
| 39 | `LowerVtcmAlloc` | `s_tir.transform` | 降低 VTCM allocation。 |
| 40 | `VerifyMemory` | `tirx.transform` | 校验是否存在 host 侧非法直接访问 device memory。 |
| 41 | `AnnotateEntryFunc` | `tirx.transform` | 推断并标记入口 `PrimFunc`。 |
| 42 | `ThreadSync("shared")` | `s_tir.transform` | 为 shared memory 的并行读写插入同步。 |
| 43 | `ThreadSync("shared.dyn")` | `s_tir.transform` | 为动态 shared memory 的并行读写插入同步。 |
| 44 | `ThreadSync("warp")` | `s_tir.transform` | 为 warp scope memory 的并行读写插入同步。 |
| 45 | `InferFragment` | `s_tir.transform` | 根据 tensor intrinsic 推断 TensorCore fragment 信息。 |
| 46 | `LowerThreadAllreduce` | `s_tir.transform` | 降低 thread allreduce。 |
| 47 | `InjectPTXAsyncCopy` | `s_tir.transform` | CUDA 上把 global-to-shared copy 改写成 async copy；仅 `tirx.use_async_copy=True` 时插入。 |
| 48 | `InjectPTXLDG32()` | `s_tir.transform` | 第二处 ldg32 注入点；仅 `tirx.ptx_ldg32=True` 时插入。 |
| 49 | `AnnotateDeviceRegions` | `tirx.transform` | 标注 host 函数中应作为 device code 的区域。 |
| 50 | `SplitHostDevice` | `tirx.transform` | 把 device region 抽成独立 device `PrimFunc`，host 侧留下调用。 |
| 51 | `MergeSharedMemoryAllocations` | `s_tir.transform` | 合并多个 TIR-level shared memory allocation；必须跟在 `SplitHostDevice` 后。 |
| 52 | `MakePackedAPI` | `tirx.transform` | 把外部函数改写成 TVM packed API，消费 `buffer_map` 并生成参数检查/unpack。 |
| 53 | `FP8StorageLegalize` | `tirx.transform` | 把 fp8 storage 合法化成等宽整数 storage。 |
| 54 | `BF16StorageLegalize` | `tirx.transform` | 把 bf16 storage 合法化成等宽整数 storage。 |
| 55 | `LowerDeviceKernelLaunch` | `tirx.transform` | 把 host 到 device kernel 的调用降低成 `tvm_call_packed`。 |

#### 10.6.3 Build 阶段模块拆分 pass

`default_s_tir_pipeline` 跑完后，`build.py::split_host_device_mods` 会继续用
`Filter` pass 把同一个 `IRModule` 拆成 host module 和 device modules。

| 顺序 | Pass | 来源 | 作用 |
| --- | --- | --- | --- |
| 1 | `tirx.transform.Filter(is_host_func)` | `python/tvm/tirx/build.py::split_host_device_mods` | 保留 host 函数，形成 `host_mod`。当前判断是 target kind 为 `llvm` 或 `c`。 |
| 2 | `tirx.transform.Filter(lambda f: not is_host_func(f))` | `python/tvm/tirx/build.py::split_host_device_mods` | 保留 device 函数，再按 device target 分组成多个 `device_mod`。 |

#### 10.6.4 Finalization pass

模块拆分后，host 和 device 分别跑 finalization pass。代码位置仍在
`python/tvm/s_tir/pipeline.py`。

host finalization：

| 顺序 | Pass | 作用 |
| --- | --- | --- |
| 1 | `tirx.transform.LowerTVMBuiltin` | 降低 host 侧 TVM builtin/runtime 调用。 |
| 2 | `tirx.transform.LowerCustomDatatypes` | 降低自定义 datatype。 |
| 3 | `tirx.transform.LowerIntrin` | 降低 target-specific intrinsic。 |

device finalization：

| 顺序 | Pass | 作用 |
| --- | --- | --- |
| 1 | `tirx.transform.LowerWarpMemory` | 降低 warp memory 访问并更新 storage scope。 |
| 2 | `tirx.transform.Simplify` | 做最后一轮设备端 IR 化简。 |
| 3 | `tirx.transform.LowerCustomDatatypes` | 降低设备端自定义 datatype。 |
| 4 | `tirx.transform.LowerIntrin` | 降低设备端 target intrinsic。 |

#### 10.6.5 Codegen 前可选 pass

`codegen_build` 在调用 `target.build.<kind>` 前还有一个条件 pass：

| 条件 | Pass | 作用 |
| --- | --- | --- |
| `tirx.disable_assert=True` | `tirx.transform.SkipAssert` | 在 host/device module codegen 前移除 `AssertStmt`。这个 pass 不属于 default pipeline 主体，而是在 `python/tvm/tirx/build.py::codegen_build` 中按需执行。 |

## 11. 预置 pipeline 中的 TIRX pass 顺序

### 11.1 `tirx_pipeline`

`python/tvm/tirx/compilation_pipeline.py` 中的 `tirx_pipeline()` 是较纯的
TIRX lowering pipeline：

```text
LowerTIRx
UnifyThreadBinding                # s_tir pass
Simplify
LowerTIRxOpaque
FlattenBuffer
BF16ComputeLegalize
NarrowDataType(32)
VectorizeLoop
UnrollLoop
Simplify
CommonSubexprElim                 # 可由 tir.disable_cse_tir 关闭
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

host finalization：

```text
LowerTVMBuiltin
LowerCustomDatatypes
LowerIntrin
```

device finalization：

```text
LowerWarpMemory
Simplify
LowerCustomDatatypes
LowerIntrin
```

### 11.2 `trn_pipeline`

Trainium pipeline 更短，且先做 TRN 私有 buffer 分配：

```text
trn.TrnPrivateBufferAlloc
trn.TrnNaiveAllocator
LowerTIRx
DecorateDeviceScope               # s_tir pass
Simplify
LowerTIRxOpaque
LoopPartition                     # s_tir pass
HoistIfThenElse                   # s_tir pass
Simplify
RemoveNoOp
AnnotateEntryFunc
AnnotateDeviceRegions
SplitHostDevice
MakePackedAPI
LowerDeviceKernelLaunch
```

### 11.3 `s_tir` pipeline 中混用的 TIRX pass

`python/tvm/s_tir/pipeline.py` 是 S-TIR backend pipeline，但它使用 TIRX 的
底层 pass 来完成 buffer、datatype、ABI、host/device 拆分和 intrinsic lowering：

```text
Simplify
FlattenBuffer
BF16ComputeLegalize
NarrowDataType(32)
VectorizeLoop
StorageRewrite
UnrollLoop
Simplify
RemoveNoOp
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
LowerTVMBuiltin
LowerCustomDatatypes
LowerIntrin
LowerWarpMemory
```

## 12. 源码里存在但不算公开 Python API 的 pass

| 名称 | 状态 | 说明 |
| --- | --- | --- |
| `LowerTIRxStripExecScope` | 内部 pass | 只在 `LowerTIRx()` 内部使用，没有注册成 `tirx.transform.*` FFI API。 |
| `RemoveAssumeInternal` | 内部 pass | `RemoveAssume()` 的第一步，之后会接 `RemoveNoOp()`。 |
| `LowerTIRxDedupCuTensorMaps` | 有实现但未公开 | `src/tirx/transform/lower_tirx_dedup_tensormap.cc` 中实现了 pass，用于把参数相同的 CUDA tensormap encode/alloca 去重，但当前没有在 `transform.h` 声明，也没有注册成 Python FFI API。 |
| `UnifiedStaticMemoryPlanner` | 仅声明 | `include/tvm/tirx/transform.h` 有声明，当前 `src/tirx` 中没有找到实现和 FFI 注册。 |
| `LowerInitBlock` | 可疑旧引用 | `python/tvm/tirx/compilation_pipeline.py` 的 `default_tir_pipeline()` 引用了 `tirx.transform.LowerInitBlock()`，但当前 TIRX Python wrapper 和 C++ `src/tirx` 中没有对应实现；S-TIR 侧有 `s_tir.transform.LowerInitBlock()`。 |

## 13. 常见配置项

这些配置通过 `PassContext(config={...})` 影响 TIRX pipeline 或具体 pass：

| 配置 | 影响 |
| --- | --- |
| `tirx.disable_vectorize` / `tir.disable_vectorize` | 控制 pipeline 中 `VectorizeLoop` 是否真正 vectorize。不同 pipeline 里历史上同时出现了 `tirx.*` 和 `tir.*` key。 |
| `tirx.disable_cse_tir` / `tir.disable_cse_tir` | 控制是否跳过 `CommonSubexprElim`。 |
| `tirx.disable_storage_rewrite` | 控制 `s_tir` pipeline 中是否跳过 `StorageRewrite`。 |
| `tirx.merge_static_smem` | 影响 `StorageRewrite` 的 shared memory 复用策略。 |
| `tirx.use_async_copy` | 在 `s_tir` pipeline 中打开 async copy 相关 lowering/injection。 |
| `tirx.ptx_ldg32` | 在 `s_tir` pipeline 中插入 `InjectPTXLDG32`。 |
| `tirx.instrument_bound_checkers` | 在 `s_tir` pipeline 中插入 bounds checker。 |
| `tirx.instrument_lwp` | 在 `s_tir` pipeline 中插入 profiling intrinsic。 |
| `tirx.enable_fast_math` | 影响 `LowerIntrin` 的 fast math lowering。 |
| `tirx.Simplify` | `Simplify` pass 的 `arith::SimplifyConfig`。 |
| `tirx.UnrollLoop` | `UnrollLoop` pass 的 unroll 策略配置。 |
| `tirx.RemoveNoOp` | `RemoveNoOp` pass 的 dataflow、rewrite step、profiler call 处理配置。 |

## 14. 一句话总图

```text
TIRX 高层 IR
  ExecScopeStmt / TilePrimitiveCall / layout buffer
    -> LowerTIRx / LowerTIRxOpaque
低层计算 IR
  flattened buffer / simplified expr / vectorized or unrolled loop
    -> dtype legalize / memory rewrite / CSE
host-device module
  AnnotateDeviceRegions / SplitHostDevice / MakePackedAPI / LowerDeviceKernelLaunch
target codegen ready IR
  LowerTVMBuiltin / LowerCustomDatatypes / LowerIntrin / LowerWarpMemory
```
