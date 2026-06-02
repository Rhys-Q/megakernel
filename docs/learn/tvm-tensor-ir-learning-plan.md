# TVM Tensor IR 设计与实现学习计划

本文是一份从零到完整链路的学习计划，基于当前仓库的 `3rdparty/tvm`。目标不是只会写几个 TVMScript kernel，而是能解释 Tensor IR 的数据结构、Python 构造路径、调度与 pass 体系、lowering pipeline、codegen/runtime 接口，并能独立追踪一个 kernel 从 Python 源码到 runtime module 的每一步。

## 0. 本仓库里的 TIR 命名

当前 `3rdparty/tvm` 不是上游 TVM 的原始目录布局。学习时要先接受这个仓库的实际分层：

- `tvm.tirx`：核心 TIR/TIRX IR 节点、buffer、expr、stmt、PrimFunc、build 入口、通用 lowering pass、codegen 前的 host/device 标注、kernel 函数生成和 module 拆分 helper。
- `tvm.s_tir`：更接近传统 TIR schedule/block 体系的层，包括 schedule primitive、SBlock 分析、meta-schedule、GPU lowering pass。
- `tvm.script`：TVMScript 通用 parser、IR builder、printer 机制，`tirx` dialect 挂在这里。
- `src/target` 与 `src/runtime`：target codegen、runtime module、device API、CUDA/OpenCL/LLVM/C host 等最终执行接口。

所以本计划里的 “TVM Tensor IR” 主要覆盖 `tirx + s_tir + tvm.script + target/runtime` 四块，而不是只读一个叫 `src/tir` 的目录。当前仓库中没有 `3rdparty/tvm/src/tir`、`include/tvm/tir`、`python/tvm/tir` 这类上游路径。

## 1. 学习目标

完成本计划后，你应该能回答这些问题：

1. TVM IR 的对象系统、FFI、`IRModule`、`PrimFunc`、`Target`、`Pass` 如何连接。
2. `@T.prim_func` 为什么不是普通 Python 函数执行，而是通过 TVMScript parser 和 IR builder 构造 IR。
3. `tirx.PrimFunc`、`Buffer`、`Var`、`BufferLoad`、`BufferStore`、`For`、`SBlock`、`ExecScopeStmt` 等节点的 C++/Python 定义在哪里。
4. S-TIR 的 block/schedule 抽象如何表达计算块、迭代域、读写 region、调度 trace。
5. 一个 kernel 经过 `tvm.tirx.build` 时，target binding、pipeline pass、`SplitHostDevice`、`split_host_device_mods`、packed API、device launch lowering 如何发生。
6. 最终 codegen 如何调用 `target.build.cuda`、`target.build.llvm`、`target.build.c` 等注册函数，并产出 `tvm.runtime.Module`。
7. 如何用测试、`mod.script()`、单 pass 调试、`PassContext` 配置去定位 IR lowering 问题。

## 2. 推荐节奏

建议按 8 周推进。每天 1 到 2 小时，周末做一次小实验和笔记整理。每周交付一份短文档或 demo，避免只读源码不产出。

| 周次 | 主题 | 主要产出 |
| --- | --- | --- |
| 第 1 周 | TVM 基础对象系统与仓库地图 | 一张模块依赖图和源码索引 |
| 第 2 周 | TVMScript parser/builder/printer | 一个 `@T.prim_func` 到 IR 的追踪笔记 |
| 第 3 周 | TIRX 核心 IR 数据模型 | Expr/Stmt/Buffer/PrimFunc 节点笔记 |
| 第 4 周 | S-TIR block 与 schedule | 一个 matmul schedule trace 实验 |
| 第 5 周 | Analysis 与 transform pass 机制 | 三个 pass 的输入/输出对比 |
| 第 6 周 | lowering pipeline | `s_tir` 与 `tirx` pipeline 全流程图 |
| 第 7 周 | codegen 与 runtime | host/device module、packed API、CUDA launch 路径笔记 |
| 第 8 周 | 综合项目 | 独立追踪 vector add 或 matmul 从 Python 到 runtime |

## 3. 准备阶段：建立读源码环境

先确认 Python 能导入本地 TVM，并能运行小测试。

```bash
cd /root/tw/megakernel/3rdparty/tvm
python - <<'PY'
import tvm
from tvm.script import tirx as T
print(tvm.__file__)
print(T)
PY
```

建议先跑窄范围测试，而不是全量测试：

```bash
cd /root/tw/megakernel/3rdparty/tvm
pytest tests/python/tirx/test_parser_printer.py -q
pytest tests/python/tirx-base/test_tir_nodes.py -q
pytest tests/python/s_tir/schedule/test_tir_schedule_split_fuse.py -q
```

读源码时优先使用这些命令：

```bash
rg "class PrimFunc|register_object|GlobalDef|target.build|LowerTIRx|MakePackedAPI" .
rg "prim_func\\(|s_tir=True|tirx.build" tests/python
```

阶段验收：

- 能说清楚本地 TVM Python 包从哪里加载。
- 能区分 `python/tvm/*` wrapper、`include/tvm/*` C++ API、`src/*` 实现、`tests/python/*` 行为样例。
- 能用 `pytest` 跑一个 TIRX 或 S-TIR 单测。

## 4. 第 1 周：TVM IR 基础

阅读入口：

- `3rdparty/tvm/include/tvm/ir/*.h`
- `3rdparty/tvm/src/ir/*`
- `3rdparty/tvm/python/tvm/ir/*`
- `3rdparty/tvm/include/tvm/target/*.h`
- `3rdparty/tvm/src/target/target.cc`
- `3rdparty/tvm/include/tvm/ir/transform.h`

重点问题：

- TVM 的 `ObjectRef` / `Object` / `Node` / FFI 注册模型是什么。
- `IRModule` 如何保存 `GlobalVar -> BaseFunc`。
- `Target` 如何用嵌套 `host` 字段表示 `cuda -host=llvm` 这类组合 target，而不是依赖旧式 CLI target string。
- `Pass`、`Sequential`、`PassContext` 的调用模型是什么。

建议实验：

1. 写一个最小 `IRModule`，打印 `mod.script()`。
2. 查一个 pass 的 Python wrapper 和 C++ 注册函数如何对应。
3. 在 Python 中构造 `tvm.target.Target("cuda", host="llvm")`，观察 `kind`、`keys`、`host` 和 `export()`。

阶段产出：

- `IRModule / PrimFunc / Target / Pass` 的关系图。
- 一份 “Python API 调用如何进入 C++ FFI” 的短笔记。

## 5. 第 2 周：TVMScript 入口与 parser/builder/printer

阅读入口：

- `3rdparty/tvm/docs/arch/tvmscript.rst`
- `3rdparty/tvm/python/tvm/script/__init__.py`
- `3rdparty/tvm/python/tvm/script/parser/core/*`
- `3rdparty/tvm/python/tvm/script/ir_builder/base.py`
- `3rdparty/tvm/src/script/ir_builder/*`
- `3rdparty/tvm/src/script/printer/*`
- `3rdparty/tvm/python/tvm/tirx/script/parser/entry.py`
- `3rdparty/tvm/python/tvm/tirx/script/parser/parser.py`
- `3rdparty/tvm/python/tvm/tirx/script/builder/*`

重点问题：

- `from tvm.script import tirx as T` 如何通过 dialect registry 定位到 `tvm.tirx.script`。
- `@T.prim_func` 如何提取 Python AST，并把 `for`、`with`、assignment 变成 IR builder 调用。
- builder frame 的 push/pop 机制如何把 Python 语法块变成 IR 节点。
- `mod.script()` 为什么能反向打印为 TVMScript。

建议实验：

```python
import tvm
from tvm.script import tirx as T

@T.prim_func(s_tir=True)
def add(a: T.handle, b: T.handle, c: T.handle):
    A = T.match_buffer(a, (128,), "float32")
    B = T.match_buffer(b, (128,), "float32")
    C = T.match_buffer(c, (128,), "float32")
    for i in T.serial(128):
        C[i] = A[i] + B[i]

mod = tvm.IRModule({"add": add})
print(mod.script())
```

阶段产出：

- 一份 “`@T.prim_func` 构造 IR 的调用栈”。
- 至少标出 parser、builder、printer 三条路径的核心文件。

## 6. 第 3 周：TIRX 核心 IR 数据模型

阅读入口：

- `3rdparty/tvm/include/tvm/tirx/function.h`
- `3rdparty/tvm/include/tvm/tirx/expr.h`
- `3rdparty/tvm/include/tvm/tirx/stmt.h`
- `3rdparty/tvm/include/tvm/tirx/buffer.h`
- `3rdparty/tvm/include/tvm/tirx/exec_scope.h`
- `3rdparty/tvm/include/tvm/tirx/layout.h`
- `3rdparty/tvm/src/tirx/ir/*.cc`
- `3rdparty/tvm/python/tvm/tirx/{function,expr,stmt,buffer,exec_scope,layout}.py`
- `3rdparty/tvm/tests/python/tirx-base/*`

重点问题：

- `PrimFunc` 的 `params`、`buffer_map`、`body`、`attrs` 各自负责什么。
- `PrimExpr` 和 `Stmt` 的节点层次如何组织。
- `Buffer` 的 shape、dtype、strides、storage scope 如何表达。
- `ExecScopeStmt` 与 `ScopeIdDef` 如何描述 `kernel / cta / warp / thread` 等执行层级。
- layout 与 tile primitive 为什么在 TIRX 中成为一等概念。

建议实验：

1. 用 Python 直接构造 `tirx.Var`、`tirx.IntImm`、`tirx.Add`、`tirx.BufferLoad`。
2. 对比 `tests/python/tirx-base/test_tir_nodes.py` 和 C++ 节点定义。
3. 打印一个 `with T.kernel(): ... with T.thread(): ...` 的 TIRX 函数，观察 `ExecScopeStmt`。

阶段产出：

- 一张 TIRX IR 节点速查表。
- 一份 “vector add 在 IR 节点层的树形结构”。

## 7. 第 4 周：S-TIR block 与 schedule

阅读入口：

- `3rdparty/tvm/include/tvm/s_tir/stmt.h`
- `3rdparty/tvm/include/tvm/tirx/stmt.h`
- `3rdparty/tvm/include/tvm/s_tir/sblock_scope.h`
- `3rdparty/tvm/include/tvm/s_tir/schedule/*.h`
- `3rdparty/tvm/src/s_tir/schedule/*`
- `3rdparty/tvm/python/tvm/s_tir/schedule/*`
- `3rdparty/tvm/tests/python/s_tir/schedule/*`
- `3rdparty/tvm/python/tvm/s_tir/tensor_intrin/*`

重点问题：

- `SBlock` / `SBlockRealize` 的节点定义为什么在 `tirx::Stmt` 体系里，而 S-TIR 在其上维护 schedule 状态。
- `StmtSRef`、`SBlockScope`、`ScheduleState` 如何维护 sref tree、依赖信息和 AST 替换。
- `Schedule`、`LoopRV`、`SBlockRV`、`Trace` 是什么。
- `split`、`fuse`、`reorder`、`bind`、`cache_read/write`、`compute_at`、`tensorize` 如何修改 IR。
- schedule primitive 的合法性检查在哪里做。

建议实验：

1. 选一个 `tests/python/s_tir/schedule/test_tir_schedule_split_fuse.py` 的 case，手动跑每一步并打印 IR。
2. 对一个 128x128 matmul 依次执行 `split`、`reorder`、`bind`，记录 `trace`。
3. 找一个失败测试，阅读它期望的错误类型和触发条件。

阶段产出：

- 一份 matmul schedule trace 笔记。
- 一张常用 schedule primitive 与源码入口对应表。

## 8. 第 5 周：analysis 与 transform pass

阅读入口：

- `3rdparty/tvm/include/tvm/tirx/analysis.h`
- `3rdparty/tvm/src/tirx/analysis/*`
- `3rdparty/tvm/include/tvm/tirx/transform.h`
- `3rdparty/tvm/src/tirx/transform/*`
- `3rdparty/tvm/include/tvm/s_tir/analysis.h`
- `3rdparty/tvm/src/s_tir/analysis/*`
- `3rdparty/tvm/include/tvm/s_tir/transform.h`
- `3rdparty/tvm/src/s_tir/transform/*`
- `3rdparty/tvm/tests/python/tirx-transform/*`
- `3rdparty/tvm/tests/python/s_tir/transform/*`

重点问题：

- analysis pass 与 transform pass 的区别是什么。
- `StmtExprMutator` / visitor / rewriter 如何遍历和替换节点。
- `Simplify`、`FlattenBuffer`、`UnifyThreadBinding`、`LowerMatchBuffer`、`LowerOpaqueBlock`、`StorageRewrite` 各自解决什么问题。
- pass 的顺序为什么重要。

建议实验：

```python
from tvm import tirx

mod1 = tirx.transform.Simplify()(mod)
mod2 = tirx.transform.FlattenBuffer()(mod1)
print(mod1.script())
print(mod2.script())
```

阶段产出：

- 选择三个 pass，记录输入 IR、输出 IR、关键源码文件、核心 rewrite 规则。
- 建立一份 “pass 调试 checklist”：目标 attr、buffer_map、storage scope、thread binding、calling_conv。

## 9. 第 6 周：lowering pipeline

阅读入口：

- `3rdparty/tvm/python/tvm/tirx/build.py`
- `3rdparty/tvm/python/tvm/tirx/compilation_pipeline.py`
- `3rdparty/tvm/python/tvm/s_tir/pipeline.py`
- `3rdparty/tvm/python/tvm/s_tir/backend/adreno/pipeline.py`
- `3rdparty/tvm/src/tirx/transform/split_host_device.cc`
- `3rdparty/tvm/src/tirx/transform/make_packed_api.cc`
- `3rdparty/tvm/src/tirx/transform/lower_device_kernel_launch.cc`

重点问题：

- `tvm.tirx.build(mod, target, pipeline)` 的控制流是什么：确定默认 target、确定 pipeline target、确定 host target、`BindTarget`、执行 pipeline、`split_host_device_mods`、finalize host/device passes、`tir_to_runtime`。
- `pipeline="default"` 当前映射到 `s_tir`；`pipeline=None` 才走 `get_default_tir_pipeline(target)` 的 target-dependent 选择。理解这两种入口的区别。
- `pipeline="s_tir"` 和 `pipeline="tirx"` 的输入假设和 pass 序列差异是什么：`s_tir` pipeline 面向 SBlock/schedule 风格 IR；`tirx` pipeline 先做 `LowerTIRx` / `LowerTIRxOpaque`，整体链更短。
- `BindTarget` 在 pipeline 前如何给函数绑定 `target` attr，并如何按 host/device 调用关系选择 full target、host target 或 without-host device target。
- `AnnotateEntryFunc`、`AnnotateDeviceRegions`、`SplitHostDevice` 如何标注入口、标出 device region，并在同一个 `IRModule` 中生成 device kernel 函数。
- `split_host_device_mods` 如何在 pipeline 之后真正拆成 `host_mod` 和按 `Target` 分组的 `device_mod_dict`。
- `MakePackedAPI` 如何把用户函数改写成 TVM packed function ABI。
- `LowerDeviceKernelLaunch` 如何在 host 侧生成设备 kernel launch 调用。

建议实验：

1. 用 `@T.prim_func(s_tir=True)` 的 vector add 走 `pipeline="s_tir"`，逐个 pass 打印 IR。
2. 另选一个 TIRX exec-scope 或 tile-primitive 风格的最小 case 走 `pipeline="tirx"`；不要把同一个 SBlock 程序当作两条 pipeline 的公平对照。
3. 在 `PassContext` 中开关对应配置，观察 pipeline 行为：`s_tir` pipeline 使用 `tirx.disable_vectorize` / `tirx.disable_cse_tir`；`tirx` pipeline 使用 `tir.disable_vectorize` / `tir.disable_cse_tir`。
4. 对 `SplitHostDevice` 前后的同一个 module 做函数列表和 attrs 对比，再对 `split_host_device_mods` 的 `host_mod` / `device_mod_dict` 做对比。

阶段产出：

- 一张完整 lowering pipeline 时序图。
- 一份 “每个关键 pass 前后 IR 形态” 的对照表，并明确哪些 pass 只是改写同一个 `IRModule`，哪些 Python helper 真正拆分 host/device modules。

## 10. 第 7 周：codegen 与 runtime

阅读入口：

- `3rdparty/tvm/python/tvm/tirx/build.py`
- `3rdparty/tvm/src/target/codegen.cc`
- `3rdparty/tvm/src/target/cuda/codegen_cuda.cc`
- `3rdparty/tvm/src/target/llvm/llvm_module.cc`
- `3rdparty/tvm/src/target/source/codegen_c_host.cc`
- `3rdparty/tvm/src/runtime/module.cc`
- `3rdparty/tvm/src/runtime/tensor.cc`
- `3rdparty/tvm/src/runtime/cuda/cuda_module.cc`
- `3rdparty/tvm/src/runtime/cuda/cuda_device_api.cc`
- `3rdparty/tvm/include/tvm/runtime/*.h`

重点问题：

- `codegen_build` 如何通过 `"target.build." + target.kind.name` 找到后端 codegen。
- host module 和 device module 如何通过 `import_module` 合并。
- `tvm.runtime.Module` 如何查找和调用 function。
- packed function 参数如何从 Python `NDArray`/标量进入 C ABI。
- CUDA module 如何保存 PTX/CUBIN，并在 runtime launch kernel。

建议实验：

1. 构造 CPU-only vector add，查看生成 runtime module 的 imported modules。
2. 如果 CUDA 可用，构造 CUDA kernel，打印 `rt_mod.imported_modules` 和 device source。
3. 阅读 `tests/python/tirx/codegen/test_codegen_cuda.py`，选择一个最小 case 追踪到 `BuildCUDA`。

阶段产出：

- 一份 host/device runtime module 生命周期说明。
- 一张 “Python 调用 compiled function -> packed API -> runtime -> device launch” 路径图。

## 11. 第 8 周：综合项目

选择一个主线 demo，全程追踪。推荐两个难度：

- 入门：vector add 或 add one。
- 进阶：128x128 matmul，包含 schedule、thread binding、shared/local memory、tensorize 或 tile primitive。

综合项目必须覆盖这些检查点：

1. Python TVMScript 源码。
2. parser/builder 后的初始 `IRModule`。
3. schedule 或 TIRX 扩展后的 IR。
4. 每个关键 lowering pass 后的 IR。
5. `SplitHostDevice` 后同一个 `IRModule` 里的函数列表和 attrs。
6. `split_host_device_mods` 后的 `host_mod` 与 `device_mod_dict`。
7. packed API 后的 host 函数签名。
8. target codegen 入口。
9. runtime module 结构。
10. 实际运行结果或无法运行时的原因。

建议最终产出：

- `docs/learn/tvm-tensor-ir-end-to-end.md`
- `example/tvm_tensor_ir/` 下的最小可运行脚本
- 一份你自己的 “TIR debug 手册”

## 12. 源码索引

核心 IR：

- `3rdparty/tvm/include/tvm/tirx/*.h`
- `3rdparty/tvm/src/tirx/ir/*`
- `3rdparty/tvm/python/tvm/tirx/*.py`

S-TIR schedule 与 SBlock 相关状态：

- `3rdparty/tvm/include/tvm/tirx/stmt.h`
- `3rdparty/tvm/include/tvm/s_tir/*.h`
- `3rdparty/tvm/include/tvm/s_tir/sblock_scope.h`
- `3rdparty/tvm/include/tvm/s_tir/schedule/*.h`
- `3rdparty/tvm/include/tvm/s_tir/schedule/state.h`
- `3rdparty/tvm/src/s_tir/schedule/*`
- `3rdparty/tvm/python/tvm/s_tir/schedule/*`

TVMScript：

- `3rdparty/tvm/docs/arch/tvmscript.rst`
- `3rdparty/tvm/python/tvm/script/*`
- `3rdparty/tvm/python/tvm/tirx/script/*`
- `3rdparty/tvm/src/script/*`

Pass 与 pipeline：

- `3rdparty/tvm/python/tvm/tirx/build.py`
- `3rdparty/tvm/python/tvm/tirx/compilation_pipeline.py`
- `3rdparty/tvm/python/tvm/s_tir/pipeline.py`
- `3rdparty/tvm/python/tvm/s_tir/backend/adreno/pipeline.py`
- `3rdparty/tvm/src/tirx/transform/*`
- `3rdparty/tvm/src/s_tir/transform/*`

Build/codegen/runtime：

- `3rdparty/tvm/src/target/codegen.cc`
- `3rdparty/tvm/src/target/cuda/codegen_cuda.cc`
- `3rdparty/tvm/src/target/llvm/llvm_module.cc`
- `3rdparty/tvm/src/target/source/codegen_c_host.cc`
- `3rdparty/tvm/src/runtime/*`

测试样例：

- `3rdparty/tvm/tests/python/tirx-base/*`
- `3rdparty/tvm/tests/python/tirx-transform/*`
- `3rdparty/tvm/tests/python/tirx/codegen/*`
- `3rdparty/tvm/tests/python/s_tir/schedule/*`
- `3rdparty/tvm/tests/python/s_tir/transform/*`

## 13. 推荐阅读顺序

如果时间有限，按这个顺序读：

1. `docs/arch/tvmscript.rst`
2. `python/tvm/tirx/script/parser/entry.py`
3. `include/tvm/tirx/function.h`
4. `include/tvm/tirx/expr.h`
5. `include/tvm/tirx/stmt.h`
6. `include/tvm/s_tir/stmt.h`
7. `include/tvm/s_tir/sblock_scope.h`
8. `include/tvm/s_tir/schedule/state.h`
9. `include/tvm/tirx/buffer.h`
10. `python/tvm/s_tir/schedule/schedule.py`
11. `python/tvm/tirx/build.py`
12. `python/tvm/s_tir/pipeline.py`
13. `python/tvm/tirx/compilation_pipeline.py`
14. `python/tvm/s_tir/backend/adreno/pipeline.py`
15. `src/tirx/transform/split_host_device.cc`
16. `src/tirx/transform/make_packed_api.cc`
17. `src/tirx/transform/lower_device_kernel_launch.cc`
18. `src/target/cuda/codegen_cuda.cc` 或 `src/target/source/codegen_c_host.cc`
19. `src/runtime/module.cc`

## 14. 学习时的固定问题模板

每读一个 IR 节点、pass 或 runtime 组件，都用同一组问题整理：

- 它解决的编译阶段问题是什么。
- 输入对象和输出对象分别是什么。
- Python API、C++ 头文件、C++ 实现、测试分别在哪里。
- 它依赖哪些 attrs、buffer 信息、target 信息。
- 它破坏或建立了哪些 invariant。
- 出错时最可能从哪些测试或 pass 开始定位。

这个模板能防止源码阅读变成散点记忆。Tensor IR 的关键不是记住所有节点，而是能把一个节点或 pass 放回完整编译链路中。
