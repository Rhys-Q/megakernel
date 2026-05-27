# TVM TIRX 设计学习笔记

这篇笔记基于当前仓库的 `3rdparty/tvm` 源码，目标是把 TIRX 从 Python 写法到 runtime 执行的链路串起来。TIRX 仍然落在 TVM 的底层 tensor IR 层，但它在传统 TIR/S-TIR 之外加入了更显式的执行层级、layout、tile primitive 和设备 kernel launch 表达方式，主要服务 GPU/加速器 kernel 的编写、lowering 和 codegen。

全文使用同一个最小 demo 贯穿。它只有一个 CUDA kernel：每个 thread 处理一个元素，把 `A[tx] + 1` 写到 `B[tx]`。

```python
import tvm
from tvm.script import tirx as Tx


@Tx.prim_func
def add_one(A: Tx.Buffer((128,), "float32"), B: Tx.Buffer((128,), "float32")):
    with Tx.kernel():
        tx = Tx.thread_id([128])
        with Tx.thread():
            B[tx] = A[tx] + Tx.float32(1.0)


mod = tvm.IRModule({"main": add_one})
rt_mod = tvm.tirx.build(mod, target="cuda", pipeline="tirx")
```

真实运行时还需要本机 TVM 编译时启用了 CUDA codegen，并且有 CUDA 设备。即使没有 CUDA，这个 demo 仍然适合作为 IR/lowering 的阅读样例。

## 1. TIRX 在 TVM 里的入口

TIRX 的 Python 命名空间在 `python/tvm/tirx/__init__.py` 里注册：

```python
tvm.script.register_dialect("tirx", "tvm.tirx.script")
```

所以 `from tvm.script import tirx as Tx` 得到的是一个 TVMScript dialect。`@Tx.prim_func` 的实现位于 `python/tvm/tirx/script/parser/entry.py`，核心流程是：

1. `prim_func` decorator 捕获 Python 函数对象和闭包变量。
2. 调用 TVMScript parser 的 `parse(func, extra_vars, ..., s_tir=s_tir)`。
3. parser 遇到 `Tx.Buffer`、`Tx.kernel()`、`Tx.thread_id()`、`Tx.thread()`、`B[tx] = ...` 时，调用 `tvm.tirx.script.builder` 下的 builder API。
4. builder 最终构造 C++/FFI 注册的 `tirx.PrimFunc`、`tirx.Buffer`、`tirx.ExecScopeStmt`、`tirx.BufferStore` 等节点。

对 demo 来说，Python 函数不是直接执行 kernel，而是在 import/定义阶段被解析成一个 `tirx.PrimFunc`：

```text
add_one: tirx.PrimFunc
  params: [A_handle, B_handle] 或等价 Buffer 参数
  buffer_map: 参数 -> Buffer((128,), float32)
  body:
    ExecScopeStmt(kernel)
      ScopeIdDef: thread_id extent = [128]
      ExecScopeStmt(thread)
        BufferStore(B, BufferLoad(A, [tx]) + 1.0, [tx])
```

这一步的重点是：`with Tx.kernel()` 和 `with Tx.thread()` 不是 Python 控制流，而是进入 IRBuilder frame。`include/tvm/tirx/script/builder/frame.h` 和 `python/tvm/tirx/script/builder/frame.py` 定义了这些 frame，退出 frame 时生成对应的 IR 节点。

## 2. TIRX IR 的核心对象

TIRX 的 IR 节点主要定义在 `include/tvm/tirx`。它们通过 TVM FFI 注册，因此 Python 侧看到的是同一批对象的 wrapper。

### PrimFunc

`include/tvm/tirx/function.h` 定义 `tirx.PrimFuncNode`，关键字段是：

- `params`：函数形参。
- `ret_type`：返回类型。
- `buffer_map`：把 handle 参数绑定到结构化 `Buffer`。
- `body`：函数体，类型是 `tirx.Stmt`。
- `attrs`：继承自 `BaseFuncNode`，后续 pass 会写入 `target`、`global_symbol`、`calling_conv` 等属性。

demo 里的 `add_one` 在进入 build 前基本就是一个 `PrimFunc`。`A`、`B` 是 `Buffer`，函数体是一段带执行层级的 stmt。

### Expr 和 Stmt

`include/tvm/tirx/expr.h` 定义表达式节点，例如：

- `Var`、`IntImm`、`FloatImm`
- `Add/Sub/Mul/Div`
- `BufferLoad`
- `Call`
- `Let`

`include/tvm/tirx/stmt.h` 定义语句节点，例如：

- `BufferStore`
- `DeclBuffer`、`AllocBuffer`
- `SeqStmt`
- `IfThenElse`
- `For`
- `Evaluate`
- `ExecScopeStmt`
- `SBlock/SBlockRealize`

demo 中 `B[tx] = A[tx] + 1` 会变成：

```text
BufferStore
  buffer: B
  indices: [tx]
  value:
    Add(
      BufferLoad(A, [tx]),
      FloatImm(float32, 1.0)
    )
```

### ExecScope 和 ScopeIdDef

TIRX 最有辨识度的部分是执行层级。`include/tvm/tirx/exec_scope.h` 定义了 `ScopeKind`：

```text
world > kernel > cluster > cta > warpgroup > warp > thread
```

以及 `ScopeIdDef`，用来表达父子 scope 之间的 ID 绑定和 extent。demo 里：

```python
with Tx.kernel():
    tx = Tx.thread_id([128])
    with Tx.thread():
        ...
```

在 IR 层等价于：

```text
ExecScopeStmt(kind=kernel)
  scope_id_def:
    thread_id extent [128]
  body:
    ExecScopeStmt(kind=thread)
      body:
        BufferStore(...)
```

`thread_id([128])` 不是普通变量定义，它是一个 scope id 定义。后面的 `LowerTIRx` 会把它解析成目标后端能理解的 thread axis，例如 CUDA 的 `threadIdx.x`。

### Layout 和 tile primitive

TIRX 还把 layout 作为一等对象，定义在 `include/tvm/tirx/layout.h`。常见节点包括：

- `TileLayout`
- `SwizzleLayout`
- `ComposeLayout`
- `Axis`
- `Iter`

这个最小 demo 没有显式 layout，因此只会用默认 buffer layout。更复杂的 TIRX kernel 可以把一个 logical tensor 映射到 lane、warp、shared memory swizzle 等物理布局上。

tile primitive 相关节点在 `include/tvm/tirx/tirx_stmt.h` 和 `python/tvm/tirx/operator/tile_primitive`。例如 `Tx.add(dst, src1, src2)` 这类高层 tile primitive 先作为 `TilePrimitiveCall` 进入 IR，再由 `TilePrimitiveDispatch` 选择具体实现并展开。demo 没有用 tile primitive，但 `LowerTIRx` pipeline 总是预留了这一步。

## 3. 从 Python 到 IRModule

`@Tx.prim_func` 的结果是单个 `PrimFunc`。TVM 编译入口习惯处理 `IRModule`，所以 build 时会把它包成 module：

```python
mod = tvm.IRModule({"main": add_one})
```

如果直接传 `PrimFunc`，`python/tvm/tirx/build.py` 也会做同样的事情：

```python
if isinstance(mod, PrimFunc):
    mod = tvm.IRModule.from_expr(mod)
```

此时 demo 的 module 只有一个函数：

```text
IRModule
  @main: tirx.PrimFunc
    attrs: 暂无 target/calling_conv，或只有用户显式写入的 attrs
    body: ExecScopeStmt(kernel -> thread -> BufferStore)
```

这一阶段还没有 host/device 拆分，也没有 packed API。它只是一个描述 kernel 语义和执行层级的 TIRX module。

## 4. `tvm.tirx.build` 的主流程

TIRX 的 build 入口在 `python/tvm/tirx/build.py`。它的流程可以按 7 步理解。

### Step 1：确定 target 和 host target

demo 调用：

```python
tvm.tirx.build(mod, target="cuda", pipeline="tirx")
```

`build` 会把 `"cuda"` 转成 `Target("cuda")`。host target 默认选 `llvm`，如果当前 TVM 没启用 LLVM，则回退到 `c`。最后 device target 会带上 host：

```text
target_to_bind = cuda -host=llvm
target_host = llvm
```

### Step 2：BindTarget

`tirx.transform.BindTarget(target_to_bind)` 会给没有 target attr 的 `PrimFunc` 绑定 target。demo 的 `main` 会得到 `target=cuda -host=llvm`。

形态变成：

```text
@main
  attrs:
    target = cuda -host=llvm
  body:
    ExecScopeStmt(kernel)
      ...
```

### Step 3：运行 TIRX pipeline

显式传 `pipeline="tirx"` 时，`build.py` 调用：

```python
pipeline, finalize_host_passes, finalize_device_passes = tvm.tirx.get_tir_pipeline("tirx")
mod = pipeline(mod)
```

`tirx_pipeline` 定义在 `python/tvm/tirx/compilation_pipeline.py`，顺序是：

```text
LowerTIRx
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

后面章节会按 demo 展开关键 pass。

### Step 4：按 target 拆出 host/device module

pipeline 内部已经产生 host 函数和 device 函数。`build.py` 的 `split_host_device_mods` 再按函数 attr 里的 target 把它们分组：

```text
host_mod: target kind 是 llvm/c 的函数
device_mod_dict:
  cuda -> CUDA device PrimFunc 集合
```

### Step 5：finalize host/device IR

`tirx_pipeline` 返回的 finalize pass 是：

```text
host:
  LowerTVMBuiltin
  LowerCustomDatatypes
  LowerIntrin

device:
  LowerWarpMemory
  Simplify
  LowerCustomDatatypes
  LowerIntrin
```

host 侧主要把 TVM builtin、intrinsic、特殊 dtype 降到 codegen 可接受的形态。device 侧额外处理 warp memory，再降 intrinsic。

### Step 6：调用目标后端 codegen

`codegen_build` 会查找全局函数：

```python
build_f_name = "target.build." + target.kind.name
bf = tvm.get_global_func(build_f_name)
```

对 demo 来说：

```text
host: target.build.llvm 或 target.build.c
device: target.build.cuda
```

### Step 7：链接 runtime module

`tir_to_runtime` 先 build host module，再把 device module import 进去：

```python
mhost = codegen_build(mhost_all, target_host)
for dev_mod in device_modules:
    mhost.import_module(dev_mod)
return mhost
```

最终 `rt_mod` 是一个 host runtime module，里面 import 了 CUDA device module。用户调用 `rt_mod["main"](...)` 时，入口在 host module；host 再通过 packed call launch CUDA kernel。

## 5. 关键 lowering pass：demo 的形态如何变化

### 5.1 LowerTIRx：执行层级变成 thread axis

`src/tirx/transform/lower_tirx.cc` 显示 `LowerTIRx` 不是单个 mutator，而是一个小 pipeline：

```text
LowerEventTensor
TilePrimitiveDispatch
LowerTIRxCleanup
LowerTIRxStripExecScope
```

对 demo 来说：

- `LowerEventTensor` 没有实际工作，因为没有 event tensor。
- `TilePrimitiveDispatch` 没有实际工作，因为没有 tile primitive。
- `LowerTIRxCleanup` 会解析 `ExecScopeStmt`、`ScopeIdDef`、layout/view 等 TIRX 专有结构。
- `LowerTIRxStripExecScope` 移除已经不需要的 `ExecScopeStmt` wrapper。

lower 前：

```text
ExecScopeStmt(kernel)
  ScopeIdDef(thread_id extent=[128])
  ExecScopeStmt(thread)
    B[tx] = A[tx] + 1
```

lower 后会接近普通底层 TIR 形态：

```text
threadIdx_x = launch_thread("threadIdx.x", 128)
B[threadIdx_x] = A[threadIdx_x] + 1
```

也就是说，TIRX 的 `thread_id([128])` 被解析为目标线程轴，`ExecScopeStmt` 自身不再保留。TIRX 的结构化执行层级在这里完成了最重要的一次语义落地。

### 5.2 Simplify / FlattenBuffer / datatype legalize

`Simplify` 做代数化简、常量折叠、条件整理。demo 很简单，主要保持 `threadIdx_x` 索引不变。

`FlattenBuffer` 会把多维 buffer 访问变成一维线性访问。demo 的 buffer 本来就是一维，所以形态基本不变：

```text
A[threadIdx_x]
B[threadIdx_x]
```

如果 demo 是二维：

```python
B[i, j] = A[i, j] + 1
```

则 flatten 后会变成类似：

```text
B[i * stride0 + j] = A[i * stride0 + j] + 1
```

`BF16ComputeLegalize`、`FP8ComputeLegalize`、storage legalize 处理特殊 dtype。demo 使用 `float32`，这些 pass 不改变核心语义。

### 5.3 VectorizeLoop / UnrollLoop / CSE

这些 pass 是传统低层 IR 优化：

- `VectorizeLoop`：把可向量化 loop 改写成 vector 表达。
- `UnrollLoop`：处理 unroll 标注和小循环展开。
- `CommonSubexprElim`：消除公共子表达式。

demo 没有显式 loop，只有 thread axis，所以这里变化很小。文章读 demo 时可以把它们理解为“保持正确性的同时整理 codegen 前的 IR”。

### 5.4 AnnotateEntryFunc

`AnnotateEntryFunc` 位于 `src/tirx/transform/primfunc_utils.cc`。它会在可推断时标记 module 的入口函数。demo 的 `main` 是单函数 module，因此会被标记为 `tirx.is_entry_func = true`；而 public `@Tx.prim_func` 的 `global_symbol` 通常已经由 parser/构造阶段保留为函数名。后续 `MakePackedAPI` 会把这个 public 入口变成 packed ABI 入口。

形态上，`main` 会带上类似：

```text
tirx.is_entry_func = true
global_symbol = "main"    # public PrimFunc 的导出符号
```

入口标记的生成条件由 pass 控制，作用是明确“这个函数是 module 的用户入口”；导出符号则决定 runtime 侧能按哪个名字找到它。

### 5.5 AnnotateDeviceRegions

`AnnotateDeviceRegions` 会把目标设备区域包在 target attr 下，给后续 host/device split 识别。

demo 的计算体会被标注为 CUDA device region：

```text
AttrStmt(attr_key="target", node=Target("cuda"))
  launch_thread("threadIdx.x", 128)
  B[threadIdx_x] = A[threadIdx_x] + 1
```

这是 `SplitHostDevice` 的输入。

### 5.6 SplitHostDevice：产生 host 入口和 device kernel

`src/tirx/transform/split_host_device.cc` 里的 `HostDeviceSplitter` 会查找 `AttrStmt(attr_key == target)`。遇到 device target 后，它做三件事：

1. 分析 device region 中用到但未在 region 内定义的变量，作为 kernel 参数。
2. 创建新的 device `PrimFunc`，target 是 `cuda`，名字通常来自入口名加 `_kernel`。
3. 在原 host 函数中，用一次对该 `GlobalVar` 的 call 替换原 device region。

demo 变成两个函数：

```text
@main
  target = cuda -host=llvm
  body:
    Evaluate(Call(@main_kernel, A_data, B_data, ...))

@main_kernel
  target = cuda
  tirx.is_global_func = true
  body:
    launch_thread("threadIdx.x", 128)
    B[threadIdx_x] = A[threadIdx_x] + 1
```

这里 `main` 还不是最终 runtime ABI，只是一个 host-side TIRX/TIR 函数；`main_kernel` 是真正的设备 kernel 函数。

### 5.7 MakePackedAPI：入口函数变成 TVM FFI ABI

`src/tirx/transform/make_packed_api.cc` 处理带 `global_symbol` 且 calling convention 仍是 default 的函数。它把普通函数签名改成 packed function 签名：

```text
(self_handle, args, num_args, result) -> int32
```

内部通过 `TVMFFIABIBuilder` 从 `args` 解码用户传入的 tensor 参数，生成 shape/type/device 检查和 buffer 声明。demo 的 `main(A, B)` 变成：

```text
@main(self_handle, args, num_args, result) -> int32
  calling_conv = kCPackedFunc
  target = llvm
  global_symbol = tvm_ffi 前缀 + "main"
  body:
    decode args[0] as A
    decode args[1] as B
    optional set_device(cuda, dev_id)
    call @main_kernel(...)
    return 0
```

这个 pass 是 runtime 能用 `rt_mod["main"](A_tvm, B_tvm)` 调用的关键。

### 5.8 LowerDeviceKernelLaunch：host call 变成 kernel launch

`src/tirx/transform/lower_device_kernel_launch.cc` 首先收集 device kernel 信息：

- target
- global symbol
- params
- thread extent，例如 `threadIdx.x = 128`
- dynamic shared memory size，如果有

然后检查 call 的 caller/callee target。如果 host 函数调用 CUDA device 函数，就不能保留普通 `Call(@main_kernel, ...)`，而要改成 packed runtime launch：

```text
tvm_call_packed(
  "main_kernel",
  A,
  B,
  128,        # threadIdx.x extent
  ...         # 其他 launch params
)
```

同时 device kernel 函数会被设置：

```text
calling_conv = kDeviceKernelLaunch
tirx.kernel_launch_params = ["threadIdx.x", ...]
global_symbol = "main_kernel"
ret_type = void
```

到这里，host 函数已经是 packed ABI，device 函数也带上了 kernel launch calling convention。

## 6. codegen 前后的最终产物

pipeline 和 finalize pass 结束后，`build.py` 会调用 `split_host_device_mods`。它按 `func.attrs["target"].kind.name` 分组：

```text
host_mod:
  @main
    target = llvm/c
    calling_conv = kCPackedFunc

device_mod_dict:
  Target("cuda"):
    @main_kernel
      target = cuda
      calling_conv = kDeviceKernelLaunch
```

然后分别 codegen：

```text
target.build.llvm(host_mod) -> host runtime.Module
target.build.cuda(device_mod) -> cuda runtime.Module
host.import_module(cuda_module)
```

最终的 `rt_mod` 不是单纯的 CUDA module，而是一个“host module + imported device module”的组合：

```text
rt_mod
  exports:
    main: PackedFunc
  imports:
    cuda module
      main_kernel: CUDA kernel
```

可以用类似方式执行：

```python
import numpy as np
import tvm

dev = tvm.device("cuda", 0)
a_np = np.arange(128, dtype="float32")
b_np = np.zeros(128, dtype="float32")
a = tvm.runtime.tensor(a_np, device=dev)
b = tvm.runtime.tensor(b_np, device=dev)

rt_mod["main"](a, b)
np.testing.assert_allclose(b.numpy(), a_np + 1)
```

调用路径是：

```text
Python rt_mod["main"](a, b)
  -> TVM PackedFunc
  -> host packed entry decodes TVMFFI args
  -> host entry calls tvm_call_packed("main_kernel", args..., launch params...)
  -> CUDA runtime module launches main_kernel<<<grid, block, shmem, stream>>>
  -> device kernel writes B[threadIdx.x]
```

## 7. `tvm.compile` 和 Relax build 如何进入 TIRX

用户可以直接调用 `tvm.tirx.build`，也可以走更上层的 `tvm.compile`。`python/tvm/driver/build_module.py` 中的 `compile` 对 TIR/IRModule 最终会调用：

```python
lib = tvm.tirx.build(mod, target, pipeline=tir_pipeline)
```

Relax VM build 也会把 lowering 后的 TIR module 交给 `tvm.tirx.build`，见 `python/tvm/relax/vm_build.py`。所以 TIRX build 是底层 codegen 入口，不只服务手写 TIRX script。

需要注意：`tvm.tirx.build` 的参数默认是 `pipeline="default"`，而 `get_tir_pipeline("default")` 会映射到 `s_tir` pipeline。想观察 TIRX 专有的 `ExecScope` lowering，应显式传：

```python
pipeline="tirx"
```

## 8. 用 demo 串起完整状态机

把上面的过程压缩成一张状态表：

| 阶段 | demo 的主要形态 |
| --- | --- |
| Python script | `@Tx.prim_func`、`with Tx.kernel()`、`tx = Tx.thread_id([128])`、`with Tx.thread()` |
| parser/build IR | `tirx.PrimFunc`，body 中有 `ExecScopeStmt(kernel/thread)` 和 `BufferStore` |
| BindTarget | `main` 带 `target=cuda -host=llvm` |
| LowerTIRx | `ExecScopeStmt` 被消费，`tx` 落成 `launch_thread("threadIdx.x", 128)` |
| AnnotateDeviceRegions | 计算体被 `AttrStmt(target=cuda)` 标记为 device region |
| SplitHostDevice | 产生 `main` host 函数和 `main_kernel` CUDA device 函数 |
| MakePackedAPI | `main` 改成 `(self_handle, args, num_args, result) -> int32` |
| LowerDeviceKernelLaunch | host 中对 `main_kernel` 的 call 改成 `tvm_call_packed` kernel launch |
| finalize | builtin/intrin/custom dtype 降成 codegen 可接受形式 |
| codegen | host 走 `target.build.llvm/c`，device 走 `target.build.cuda` |
| runtime | host module import CUDA module，`rt_mod["main"]` 启动 device kernel |

## 9. 设计取舍总结

TIRX 的核心设计是把 GPU kernel 作者关心的层级和 layout 明确建模在 IR 里，而不是一开始就把所有东西压平成普通 loop/thread attr。

这种设计带来几个直接好处：

- Python 侧写法接近 kernel 的执行结构：`kernel/cta/warp/thread` 是显式 scope。
- `ScopeIdDef` 把 thread/warp/cta ID 的 extent 和父子关系作为 IR 信息保存，方便 verifier、layout、lowering 使用。
- layout 是一等对象，复杂 shared/local/tile 布局可以在 lowering 前保持结构化。
- tile primitive 可以先表达“做什么”，再由 `TilePrimitiveDispatch` 决定“怎么展开”。
- build 后半段仍复用 TVM 成熟的 host/device module、packed ABI、`target.build.*` 和 runtime module import 机制。

代价是 lowering pipeline 更长，且需要在 TIRX 专有 IR 和传统 codegen 可接受 IR 之间做明确边界。这个边界大致就在 `LowerTIRx` 到 `LowerTIRxOpaque/FlattenBuffer` 一带：前面保留 TIRX 的结构化语义，后面逐步变成普通 target codegen 能处理的底层 IR。

如果只记一条主线，可以记成：

```text
Tx script
  -> tirx.PrimFunc / IRModule
  -> BindTarget
  -> LowerTIRx: scope/layout/tile primitive 落地
  -> SplitHostDevice: host 入口 + device kernel
  -> MakePackedAPI + LowerDeviceKernelLaunch
  -> target.build.host + target.build.device
  -> host runtime module import device runtime module
```

这个主线就是 demo 从一段 Python TIRX 函数变成可调用 runtime module 的完整路径。
