# 第 3 周：TIRX 核心 IR 数据模型

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

# TIRX IR 节点

一句话模型：`IRModule` 保存 `GlobalVar -> BaseFunc`，TIRX 的核心函数是
`tirx.PrimFunc`；`PrimFunc.body` 是一棵 `tirx.Stmt` 树，`Stmt` 中嵌
`tirx.PrimExpr`，buffer、layout、exec scope 作为一等对象挂在函数、语句和表达式上。

```text
IRModule
  functions: GlobalVar -> BaseFunc
    GlobalVar("main") -> tirx.PrimFunc
      params: [tirx.Var, ...]
      buffer_map: {tirx.Var: tirx.Buffer}
      attrs: DictAttrs
      body: tirx.Stmt
        SeqStmt / For / IfThenElse / ExecScopeStmt / BufferStore / ...
          PrimExpr: Var / IntImm / Add / BufferLoad / Call / ...
          Buffer: shape / dtype / strides / scope / layout
          ExecScope: kind / scope_id_def
          Layout: TileLayout / SwizzleLayout / ComposeLayout
```

## 顶层函数节点

| 节点 | 定义位置 | 关键字段 | 作用 |
| --- | --- | --- | --- |
| `tirx.PrimFunc` | `include/tvm/tirx/function.h` | `params`, `ret_type`, `buffer_map`, `body`, `attrs` | TIRX 的低层函数单位，也是 pass、build、codegen 的主要处理对象。 |
| `tirx.TensorIntrin` | `include/tvm/tirx/function.h` | `desc`, `impl` | tensorization 用的 intrin 描述和实现。 |

`PrimFunc` 字段的职责：

- `params`：函数形参，类型是 `Array<tirx::Var>`。如果参数代表 buffer handle，仍然先是一个 `Var`。
- `buffer_map`：把参数 `Var` 映射到结构化 `Buffer`。它表达 shape、dtype、stride、layout 等约束，也让 body 能直接使用 `BufferLoad/BufferStore`。
- `body`：函数体，类型是 `tirx.Stmt`。这是大部分优化和 lowering 访问的主体。
- `attrs`：继承自 `BaseFuncNode`，保存 `global_symbol`、`target`、`s_tir`、`tirx.is_entry_func` 等函数级元信息。
- `ret_type`：返回类型，绝大多数 kernel 是 `VoidType()`。

## 变量与迭代节点

| 节点 | 基类 | 关键字段 | 作用 |
| --- | --- | --- | --- |
| `tirx.Var` | `PrimExpr` | `name_hint`, `type_annotation`, `dtype` | 符号变量。变量按对象地址区分身份，同名不等价。 |
| `tirx.SizeVar` | `Var` | 继承 `Var` | 表示非负 shape/index size。 |
| `tirx.IterVar` | `PrimExprConvertible` | `dom`, `var`, `iter_type`, `thread_tag` | schedule/block/reduce/thread binding 中的迭代变量。 |

`IterVarType` 常见值：

| 值 | 含义 |
| --- | --- |
| `kDataPar` | 普通 data parallel 轴。 |
| `kThreadIndex` | 已经绑定到线程索引的轴。 |
| `kCommReduce` | 规约轴。 |
| `kOrdered` | 有 loop-carried dependency 的有序轴。 |
| `kOpaque` | opaque/external/composite op 轴。 |
| `kUnrolled`, `kVectorized`, `kParallelized`, `kTensorized` | schedule 后附加的执行属性。 |

## PrimExpr 节点

`tirx.PrimExpr` 表示标量/向量表达式，节点主要定义在
`include/tvm/tirx/expr.h` 和 `include/tvm/tirx/var.h`。

| 类别 | 节点 | 关键字段 | 说明 |
| --- | --- | --- | --- |
| 常量 | `IntImm`, `FloatImm`, `StringImm` | `value` | `IntImm/FloatImm` 复用 TVM core 节点；`StringImm` 常用于 assert message。 |
| 变量 | `Var`, `SizeVar` | `name_hint`, `type_annotation` | 自由变量、形参、loop var、scope id 都会落成 `Var`。 |
| 类型转换 | `Cast` | `value`, `dtype` | 表达 dtype cast。 |
| 算术二元 | `Add`, `Sub`, `Mul`, `Div`, `Mod`, `FloorDiv`, `FloorMod`, `Min`, `Max` | `a`, `b` | 普通算术表达式。 |
| 比较 | `EQ`, `NE`, `LT`, `LE`, `GT`, `GE` | `a`, `b` | 结果是 bool 表达式。 |
| 逻辑 | `And`, `Or`, `Not` | `a`, `b` 或 `a` | 条件组合。 |
| 条件选择 | `Select` | `condition`, `true_value`, `false_value` | 表达式级选择；不能用来保护越界访问。 |
| Buffer 读 | `BufferLoad` | `buffer`, `indices`, `predicate` | `A[i]` 或 masked vector load。 |
| Producer 读 | `ProducerLoad` | `producer`, `indices` | 高层 DSL 过渡节点，合法 TIR PrimFunc 里应先 lowering 掉。 |
| 向量 | `Ramp`, `Broadcast`, `Shuffle` | `base/stride/lanes`, `value/lanes`, `vectors/indices` | 构造向量 index、广播值、重排向量。 |
| let | `Let` | `var`, `value`, `body` | 表达式级绑定。 |
| 调用 | `Call` | `dtype`, `op`, `args`, `annotations` | 调 `Op`、`GlobalVar` 或 intrinsic；target-specific 信息可进 `annotations`。 |
| 规约 | `CommReducer`, `Reduce` | reducer 的 `lhs/rhs/result/identity_element`；reduce 的 `source/init/axis/condition/value_index` | 表达 commutative reduction。 |

表达式节点有两个容易混淆的点：

- `BufferLoad` 是 `PrimExpr`，因为读 buffer 产生一个值；`BufferStore` 是 `Stmt`，因为写 buffer 是副作用。
- `Call` 本身是表达式；如果只是为了执行有副作用的 call，需要外面包一层 `Evaluate(Call(...))`。

## Stmt 节点

`tirx.Stmt` 是函数体结构，定义在 `include/tvm/tirx/stmt.h`；
tile primitive 额外定义在 `include/tvm/tirx/tirx_stmt.h`。

| 类别 | 节点 | 关键字段 | 说明 |
| --- | --- | --- | --- |
| 绑定 | `Bind` | `var`, `value` | 在当前 enclosing scope 绑定变量，没有 body，后续语句可见。 |
| 属性 | `AttrStmt` | `node`, `attr_key`, `value`, `body` | 给一段 body 附加属性，常用于 target、device scope、pragma 等。 |
| 断言 | `AssertStmt` | `condition`, `error_kind`, `message_parts` | 运行时检查或 lowering 后保留的约束。 |
| Buffer 写 | `BufferStore` | `buffer`, `value`, `indices`, `predicate` | `B[i] = v` 或 masked store。 |
| Buffer 声明 | `DeclBuffer` | `buffer` | 声明一个可在 body 使用的 buffer。 |
| Buffer 分配 | `AllocBuffer` | `buffer`, `annotations` | 分配局部 buffer；storage scope 通常从 buffer data type/scope 推出。 |
| 顺序 | `SeqStmt` | `seq` | 多条语句的顺序容器；`SeqStmt::Flatten` 会去掉 no-op 并展开嵌套 sequence。 |
| 表达式语句 | `Evaluate` | `value` | 把表达式当语句执行，常包副作用 call；`Evaluate(0)` 常作 no-op。 |
| 分支 | `IfThenElse` | `condition`, `then_case`, `else_case` | 语句级分支。 |
| 循环 | `For` | `loop_var`, `min`, `extent`, `kind`, `body`, `thread_binding`, `annotations`, `step` | serial/parallel/vectorize/unroll/thread-binding loop。 |
| while | `While` | `condition`, `body` | while loop。 |
| 控制流 | `Break`, `Continue` | 无核心字段 | loop control。 |
| block | `SBlock` | `iter_vars`, `reads`, `writes`, `name_hint`, `alloc_buffers`, `match_buffers`, `annotations`, `init`, `body` | s_tir schedule block 的基本调度单元。 |
| block realize | `SBlockRealize` | `iter_values`, `predicate`, `block` | 给 block 的 iter vars 绑定实际值并执行。 |
| 执行层级 | `ExecScopeStmt` | `exec_scope`, `body` | TIRX 专有，表示 `kernel/cta/warp/thread` 等硬件执行层级。 |
| tile primitive | `TilePrimitiveCall` | `op`, `args`, `workspace`, `config`, `dispatch` | 高层 tile primitive 调用，后续由 dispatch/lowering 展开。 |

`ForKind`：

| 值 | 含义 |
| --- | --- |
| `kSerial` | 普通串行循环。 |
| `kParallel` | CPU parallel loop。 |
| `kVectorized` | 待 vectorize 的 loop。 |
| `kUnrolled` | 必须 unroll 的 loop。 |
| `kThreadBinding` | loop var 绑定到 thread context，后续 lowering 中可移除 loop。 |

## Buffer 与区域节点

`Buffer` 是 TIRX 表达内存对象的核心，定义在
`include/tvm/tirx/buffer.h`；区域相关节点在 `stmt.h`。

| 节点 | 基类 | 关键字段 | 作用 |
| --- | --- | --- | --- |
| `Buffer` | `Object` | `data`, `dtype`, `shape`, `strides`, `elem_offset`, `name`, `data_alignment`, `offset_factor`, `buffer_type`, `axis_separators`, `layout`, `allocated_addr` | 符号化 n 维内存对象。 |
| `BufferRegion` | `PrimExprConvertible` | `buffer`, `region` | 描述 buffer 的一段区域，常用于 `reads/writes`。 |
| `MatchBufferRegion` | `Object` | `buffer`, `source` | 约束 source region 可以 remap 到 target buffer。 |
| `DataProducer` | `PrimExprConvertible` | virtual `GetShape/GetDataType/GetNameHint` | 高层 DSL 生产者抽象，合法 TIR PrimFunc 中一般不应残留。 |

`Buffer` 的几个字段含义：

- `data`：底层数据指针变量，通常是函数参数或 allocation 的 data var。
- `shape`：按用户访问维度表达的 logical shape。
- `strides`：为空表示 compact/contiguous；非空表示显式 stride。
- `elem_offset`：以 dtype element 为单位的偏移。
- `axis_separators`：多维逻辑轴 flatten 成输出轴时的分隔信息。
- `layout`：TIRX 扩展，把 tile/swizzle/layout 作为 buffer 的一等属性。
- `allocated_addr`：某些目标上 allocation 地址可能是多维地址，例如 bank/offset。

## ExecScope 节点

TIRX 最有辨识度的结构是把硬件执行层级显式放进 IR。
相关定义在 `include/tvm/tirx/exec_scope.h`，语句 wrapper 是 `ExecScopeStmt`。

| 节点/枚举 | 关键字段/值 | 作用 |
| --- | --- | --- |
| `ScopeKind` | `kWorld`, `kKernel`, `kCluster`, `kCta`, `kWarpgroup`, `kWarp`, `kThread` | 从粗到细的执行层级。 |
| `ScopeBinding` | `kKernelCluster`, `kKernelCta`, `kClusterCta`, `kCtaWarpgroup`, `kCtaWarp`, `kWarpgroupWarp`, `kWarpThread`, `kCtaThread`, `kWarpgroupThread`, `kClusterCtaPair` | 父 scope 到子 scope 的 id 绑定关系。 |
| `ScopeIdDef` | `def_ids`, `extents`, `scope`, `preferred_extents` | 定义一个或多个 scope id 变量及其 extent。`extents=None` 表示 deferred，后续 verifier/lowering 推断。 |
| `ExecScope` | `kind`, `scope_id_def` | 一个执行层级对象，例如 `kernel` 或 `thread`。 |
| `ExecScopeStmt` | `exec_scope`, `body` | 把 body 放进某个 execution scope。 |

典型结构：

```text
ExecScopeStmt(exec_scope=ExecScope(kind=kKernel, scope_id_def=[...]))
  ExecScopeStmt(exec_scope=ExecScope(kind=kThread, scope_id_def=[...]))
    BufferStore(...)
```

在 parser 阶段，`with Tx.kernel():`、`with Tx.thread():` 等 Python 块会进入
builder frame；frame 退出时构造 `ExecScopeStmt`。在 `LowerTIRx` 中，
这些结构会被消费，scope id 逐步落成目标后端可理解的 launch/thread 语义。

## Layout 节点

Layout 定义在 `include/tvm/tirx/layout.h`。它不是普通注释，而是直接参与
buffer 表达、tile primitive dispatch、layout transform 和后端 lowering。

| 节点 | 关键字段 | 作用 |
| --- | --- | --- |
| `Layout` | 抽象接口：`CompatibleWithShape`, `Apply`, `Canonicalize`, `Tile`, `Slice`, `DirectSum` 等 | layout 基类。 |
| `Axis` | `name`，以及 registry attrs：scope/subscope/fuser/splitter | layout 坐标轴，既可表示 memory axis，也可表示 thread axis。 |
| `Iter` | `extent`, `stride`, `axis` | layout 中某个 axis 的迭代片段。 |
| `TileLayout` | `shard`, `replica`, `offset` | tile/shard/replica 映射；能判断 memory/thread axis、scope pair。 |
| `SwizzleLayout` | `per_element`, `swizzle_len`, `atom_len`, `swizzle_inner` | bank/swizzle 类布局变换。 |
| `ComposeLayout` | `swizzle`, `tile_layout` | swizzle 和 tile layout 的组合。 |

可以把 layout 理解为“index -> physical/thread/memory axes”的映射对象：

```text
logical coord
  -> Layout.Apply(...)
  -> {axis_name: mapped_expr}
```

传统 TIR 里 layout 往往藏在 index 算式或 schedule 约定里；TIRX 把它作为节点保存，
这样 tile primitive 和后端 lowering 可以直接读 layout 语义。

## TilePrimitiveCall 节点

`TilePrimitiveCall` 定义在 `include/tvm/tirx/tirx_stmt.h`，是 TIRX 高层 tile
primitive 的过渡节点。

| 字段 | 作用 |
| --- | --- |
| `op` | 对应的 `tvm::Op`，例如某个 `tirx.*` tile primitive。 |
| `args` | operator 参数，允许 buffer、expr、layout、普通配置对象等受控类型。 |
| `workspace` | primitive 需要的预分配临时 buffer。 |
| `config` | dispatcher/scheduler 使用的配置。 |
| `dispatch` | 可选的 dispatch variant 名称。 |

它不是最终低层 codegen 形态。正常流程是先保留高层语义，后续
`TilePrimitiveDispatch` 或相关 lowering pass 根据 target、layout、scope 选择实现并展开。

## Vector Add 的 IR 树

假设 TVMScript 形态类似：

```python
from tvm.script import tirx as Tx

@Tx.prim_func
def vadd(A: Tx.Buffer((128,), "float32"),
         B: Tx.Buffer((128,), "float32"),
         C: Tx.Buffer((128,), "float32")):
    with Tx.kernel():
        tx = Tx.thread_id([128])
        with Tx.thread():
            C[tx] = A[tx] + B[tx]
```

进入 build 前，核心 IR 可以按下面理解：

```text
IRModule
  functions
    GlobalVar("vadd")
      -> PrimFunc
           params = [A_handle, B_handle, C_handle]
           buffer_map = {
             A_handle: Buffer(name="A", shape=[128], dtype=float32),
             B_handle: Buffer(name="B", shape=[128], dtype=float32),
             C_handle: Buffer(name="C", shape=[128], dtype=float32),
           }
           attrs = {global_symbol="vadd", ...}
           body =
             ExecScopeStmt(
               exec_scope=ExecScope(kind=kKernel, scope_id_def=[...]),
               body=
                 ExecScopeStmt(
                   exec_scope=ExecScope(
                     kind=kThread,
                     scope_id_def=[
                       ScopeIdDef(def_ids=[tx], extents=[128], scope=...)
                     ],
                   ),
                   body=
                     BufferStore(
                       buffer=C,
                       indices=[Var("tx")],
                       value=
                         Add(
                           BufferLoad(buffer=A, indices=[Var("tx")]),
                           BufferLoad(buffer=B, indices=[Var("tx")]),
                         ),
                     ),
                 ),
             )
```

这棵树里最核心的几层是：

- `PrimFunc` 负责函数边界、参数和 buffer 约束。
- `ExecScopeStmt` 负责硬件执行层级。
- `ScopeIdDef` 负责把 `tx` 这类 thread/scope id 变量和 extent 绑定起来。
- `BufferStore` 是唯一的写副作用。
- `Add(BufferLoad(A), BufferLoad(B))` 是纯表达式子树。

## 阅读源码时的定位顺序

1. 先看 `include/tvm/tirx/function.h`，理解 `PrimFunc` 如何把参数、buffer 和 body 连起来。
2. 再看 `include/tvm/tirx/stmt.h`，因为函数体首先是一棵 `Stmt` 树。
3. 遇到表达式再跳到 `include/tvm/tirx/expr.h` 和 `include/tvm/tirx/var.h`。
4. 遇到内存对象看 `include/tvm/tirx/buffer.h`。
5. 遇到 `with Tx.kernel/thread/cta/warp` 相关结构看 `include/tvm/tirx/exec_scope.h`。
6. 遇到 tile/layout 相关字段看 `include/tvm/tirx/layout.h` 和 `include/tvm/tirx/tirx_stmt.h`。

一个实用判断标准：

- “产生值”的通常是 `PrimExpr`，例如 `BufferLoad`、`Add`、`Call`。
- “产生副作用或控制流”的通常是 `Stmt`，例如 `BufferStore`、`For`、`IfThenElse`、`ExecScopeStmt`。
- “描述内存/执行/布局语义”的通常是辅助对象，例如 `Buffer`、`ExecScope`、`ScopeIdDef`、`Layout`。
