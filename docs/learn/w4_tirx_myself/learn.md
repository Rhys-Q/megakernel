目录在3rdparty/tvm/python/tvm/tirx

本目录学习文档

- frontend.md: TIRX TVMScript 前端与 parser/builder 学习笔记。
- ir.md: TIRX IR 对象模型学习笔记。
- tirx_passes.md: TIRX pass 和 build pipeline 学习笔记。
- codegen.md: TIRX codegen 原理与实现学习笔记。

核心顶层文件

- 3rdparty/tvm/python/tvm/tirx/__init__.py: tvm.tirx 入口，注册 tirx TVMScript dialect，导出 Expr/Stmt/Buffer/Layout/Transform/Analysis/Build 等公共 API。
- _ffi_api.py: 初始化 tirx FFI 命名空间，Python 调 C++ 的桥。
- expr.py: TIRX 表达式节点和运算符重载，如 Var、IntImm、Cast、Add/Sub/Mul、BufferLoad、Call、Reduce。
- stmt.py: 语句 IR 节点，如 For、While、Bind、AllocBuffer、SBlock、TilePrimitiveCall、ExecScopeStmt。
- buffer.py: Buffer、decl_buffer、buffer 切片/访问/布局相关封装。
- function.py: PrimFunc、TensorIntrin、IndexMap。
- op.py: tirx.* builtin/op wrapper，包含数学函数、外部调用、PTX/CUDA intrinsic 用户侧封装。
- layout.py: TIRX layout 系统，含 Axis、Iter、TileLayout、SwizzleLayout、ComposeLayout，以及 S[...] / R[...] 这类 layout DSL。
- exec_scope.py: 执行层级定义，kernel/cluster/cta/warpgroup/warp/thread。
- exec_context.py: Python 侧执行 scope/lane active-set 推导工具，偏分析/建模。
- predicate.py: 用 Python callable 构造可应用到索引的 predicate。
- generic.py: 基础算子分发，如 add/subtract/multiply/cast。
- functor.py、expr_functor.py、stmt_functor.py: IR visitor/mutator，既有 Python 实现也有 FFI 互操作。
- build.py: tvm.tirx.build，负责绑定 target、跑 pipeline、拆 host/device module、调用 target.build.*。
- compilation_pipeline.py: 预置 lowering pipeline：default/s_tir、tirx、trn，以及 host/device finalization passes。
- bench.py: CUDA/Triton/Proton benchmark 工具，包含 L2 flush、事件计时、Perfetto trace 导出等。

analysis / transform

- analysis/analysis.py: 分析与校验 API，包含 expr_deep_equal、verify_ssa、verify_memory、undefined_vars、verify_well_formed。
- analysis/_ffi_api.py: analysis FFI 桥。
- transform/transform.py: TIRX pass wrapper，如 Simplify、FlattenBuffer、LowerTIRx、LowerIntrin、SplitHostDevice、MakePackedAPI、BindTarget。
- transform/function_pass.py: Python 自定义 PrimFuncPass 装饰器机制。
- transform/common.py: transform 辅助 visitor/mutator，比如 buffer 替换、kernel replace point 搜索。
- transform/trn/*: Trainium 专用 pass，目前主要是 private buffer 分配和 naive allocator。

script

- script/__init__.py: tvm.script.tirx 的公共入口，导出 parser/builder，并支持动态生成 TilePrimitiveCall。
- script/parser/entry.py: @Tx.prim_func、inline、macro、Buffer/Ptr proxy。
- script/parser/parser.py: Python AST 到 TIRX IR 的解析逻辑，处理 for/while/if/with/assign/return 等。
- script/parser/operation.py: 为 PrimExpr、IterVar 注册脚本解析时的运算符。
- script/builder/ir.py: TVMScript builder 主体，提供 buffer、match_buffer、sblock、kernel/cta/warp/thread、loop、dtype 构造、CUDA intrinsic namespace。
- script/builder/tirx.py: tile primitive 的脚本 API，如 copy、copy_async、gemm、gemm_async、sum/max/min、compose_op。
- script/builder/frame.py: IRBuilder frame 类型。
- script/builder/external_kernel.py、triton.py: 外部 kernel/Triton kernel 调用封装。
- script/builder/utils.py、tmem_pool.py: builder 辅助与 TMEM pool 入口。

lang

- lang/alloc_pool.py: TMEM/SMEM pool 抽象，负责分配 shared/tensor memory buffer。
- lang/pipeline.py: pipeline/ring buffer/mbarrier 抽象，服务异步 copy 和 producer-consumer pipeline。
- lang/tile_scheduler.py: 多种 tile scheduler，如 persistent 2D、group-major、FlashAttention scheduler。
- lang/smem_desc.py: shared memory descriptor helper。
- lang/warp_role.py: warp/warpgroup 角色划分工具。

operator

- operator/tile_primitive/ops.py: 定义 TIRX tile primitive op 类，如 Add/Sub/Mul、Copy、CopyAsync、Gemm、GemmAsync、Reduce、PermuteDims。
- operator/tile_primitive/dispatcher.py: 调度注册和选择逻辑，按 op、target、predicate、priority 选择实现。
- operator/tile_primitive/dispatch_context.py: dispatch 时携带 target、exec scope、launch params、buffer workspace 等上下文。
- operator/tile_primitive/cuda/*: CUDA 侧 tile primitive 实现，覆盖 scalar/vector/collective copy、TMA/cp.async/tcgen05 copy、elementwise、reduction、Blackwell tcgen05 GEMM async、permute dims。
- operator/tile_primitive/trn/*: Trainium 侧 tile primitive 实现，覆盖 unary/binary/reduction/gemm/copy/select/compose_op 和 instruction generation。
- operator/intrinsics/_common.py: intrinsic 参数枚举的单一来源，比如 fence scope、cp.async cache hint、tcgen05 shape。
- operator/intrinsics/_schema.py: 动态注册 device intrinsic helper/codegen 的通用机制。
- operator/intrinsics/cuda/*: CUDA intrinsic codegen registry 和具体 codegen，覆盖 mma/wgmma/tcgen05、cp_async/TMA、memory load/store/atomic、sync/barrier/fence、math、NVSHMEM、header generator。

其它

- backend/adreno: 目前基本是 Adreno backend namespace 占位；实际 Adreno pipeline 主要在 tvm/s_tir/backend/adreno 侧注册。
- __pycache__: Python 缓存目录，可忽略。

一个需要注意的点：tirx.compilation_pipeline 的默认 pipeline 依赖 tvm.s_tir.pipeline 往 PIPELINE_MAP 注入 "s_tir"，Adreno 也类似依赖 s_tir.backend.adreno.pipeline 注册；所以单独看 tirx/compilation_pipeline.py 会觉得 key 不全，这是跨包注册设计。
