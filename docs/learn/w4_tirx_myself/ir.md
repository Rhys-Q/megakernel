# 从 TVMScript 走进 TIRX IR：一份基于 TVM 源码的学习地图

前面如果已经学过 TVMScript 前端，会很容易形成一个错觉：TIRX
就是一套 Python DSL，`@T.prim_func`、`T.grid`、`T.sblock`、`T.buffer_store`
这些 API 把 Python 代码“翻译”成 kernel。

继续往下看 TIRX IR 时，需要把视角切换掉。TVMScript 只是入口，真正被
TVM pass、lowering、codegen 消费的是 `tvm.tirx` 这一套 IR 对象：

```text
TVMScript
  Python-like DSL, parser, builder frame

TIRX IR
  PrimFunc / Stmt / PrimExpr / Buffer / ExecScope / Layout / TilePrimitiveCall

Lowering pipeline
  tirx.transform.* passes consume and rewrite these IR nodes
```

本文基于当前仓库 `3rdparty/tvm` 的实际代码，目标不是把每个节点 API
背一遍，而是搭一个适合继续读源码的学习框架。

## 1. 先建立总图

TIRX IR 的核心结构可以压缩成一句话：

```text
IRModule
  GlobalVar -> tirx.PrimFunc
    params / buffer_map / attrs / body
      body: tirx.Stmt tree
        Stmt 中嵌 PrimExpr
        Buffer / Layout / ExecScope 作为一等对象参与 IR
```

也就是：

```text
tvm.IRModule
  functions:
    "main" -> tirx.PrimFunc
      params: Array[tirx.Var]
      buffer_map: Map[tirx.Var, tirx.Buffer]
      body: tirx.Stmt
        SeqStmt
        For
        SBlockRealize
          SBlock
            BufferStore
              BufferLoad
              Add
              FloatImm
        ExecScopeStmt
        TilePrimitiveCall
      attrs: target / global_symbol / calling_conv / tirx metadata
```

源码入口建议先看这几组文件：

| 主题 | C++ 定义 | Python wrapper |
| --- | --- | --- |
| 函数 | `include/tvm/tirx/function.h`, `src/tirx/ir/function.cc` | `python/tvm/tirx/function.py` |
| 表达式 | `include/tvm/tirx/expr.h`, `include/tvm/tirx/var.h`, `src/tirx/ir/expr.cc` | `python/tvm/tirx/expr.py` |
| 语句 | `include/tvm/tirx/stmt.h`, `src/tirx/ir/stmt.cc` | `python/tvm/tirx/stmt.py` |
| Buffer | `include/tvm/tirx/buffer.h`, `src/tirx/ir/buffer.cc` | `python/tvm/tirx/buffer.py` |
| 执行层级 | `include/tvm/tirx/exec_scope.h`, `src/tirx/ir/exec_scope.cc` | `python/tvm/tirx/exec_scope.py` |
| Layout | `include/tvm/tirx/layout.h`, `src/tirx/ir/layout/*.cc` | `python/tvm/tirx/layout.py` |
| Tile primitive | `include/tvm/tirx/tirx_stmt.h`, `src/tirx/ir/tirx_stmt.cc` | `python/tvm/tirx/operator/*` |
| IR 遍历/改写 | `include/tvm/tirx/stmt_functor.h`, `include/tvm/tirx/expr_functor.h` | `python/tvm/tirx/stmt_functor.py`, `python/tvm/tirx/expr_functor.py` |

## 2. 从一个小例子看 IR 形状

先用一个普通 s_tir 风格例子，因为它最接近你已经学过的 TVMScript：

```python
from tvm.script import tirx as T


@T.prim_func(s_tir=True)
def add_one(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")) -> None:
    for i in range(128):
        B[i] = A[i] + T.float32(1.0)
```

前端 parser 大致会构造出：

```text
tirx.PrimFunc
  params:
    A_handle
    B_handle
  buffer_map:
    A_handle -> Buffer(name="A", shape=[128], dtype=float32)
    B_handle -> Buffer(name="B", shape=[128], dtype=float32)
  body:
    For(loop_var=i, min=0, extent=128, kind=serial)
      BufferStore(
        buffer=B,
        indices=[i],
        value=Add(
          BufferLoad(buffer=A, indices=[i]),
          FloatImm(float32, 1.0)
        )
      )
```

注意两个边界：

- `A` 和 `B` 在 Python 源码里看起来是函数参数，但 IR 里通常是
  handle 参数加 `buffer_map` 里的结构化 buffer。
- `B[i] = ...` 不是 Python assignment IR，而是 `tirx.BufferStore`。
  `A[i]` 是 `tirx.BufferLoad`，它属于表达式，因为读 buffer 产生值。

如果写成更 TIRX 的执行层级风格：

```python
@T.prim_func
def add_one(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
    with T.kernel():
        tx = T.thread_id([128])
        with T.thread():
            B[tx] = A[tx] + T.float32(1.0)
```

函数体的核心形状会变成：

```text
ExecScopeStmt(kind=kernel, scope_id_def=[thread extent 128])
  ExecScopeStmt(kind=thread)
    BufferStore(...)
```

这就是 TIRX 相比传统 TIR 更显眼的地方：硬件执行层级被显式放进 IR。

## 3. PrimFunc：TIRX 函数的最小单位

`PrimFunc` 定义在 `include/tvm/tirx/function.h`：

```cpp
class PrimFuncNode : public BaseFuncNode {
 public:
  ffi::Array<tirx::Var> params;
  Type ret_type;
  ffi::Map<tirx::Var, Buffer> buffer_map;
  tirx::Stmt body;
};
```

它是后续 pass 和 build 的主要处理单位。可以把字段职责记成：

| 字段 | 作用 |
| --- | --- |
| `params` | 函数 ABI 层面的参数，类型是 `tirx.Var`。buffer 参数也先是 handle var。 |
| `buffer_map` | 把 handle var 解释成结构化 `Buffer`，携带 shape、dtype、stride、layout 等信息。 |
| `body` | 函数体，一棵 `tirx.Stmt` 树。 |
| `ret_type` | 返回类型，多数 kernel 是 `VoidType()`。 |
| `attrs` | 来自 `BaseFuncNode`，保存 `target`、`global_symbol`、`calling_conv`、`tirx.is_entry_func` 等。 |

`buffer_map` 很重要。源码注释里说它不只是语法糖，而是参数 unpacking 和
constraint checking 的一等表达。比如两个 buffer 都写 shape `[m, n]`，
那么 `m/n` 第一次出现时定义 shape 变量，多次出现时可以转成运行时约束。

Python wrapper 在 `python/tvm/tirx/function.py`。如果直接用 Python 构造
`PrimFunc(params=[Buffer(...)])`，wrapper 会把 `Buffer` 自动展开成 handle
`Var` 并写入 `buffer_map`：

```python
if isinstance(x, Buffer):
    var = Var(x.name, dtype="handle")
    param_list.append(var)
    buffer_map[var] = x
```

学习 `PrimFunc` 时要重点看两条链：

```text
TVMScript parser:
  T.arg(name, ann) -> PrimFuncFrame.args / buffer_map

手写 IR:
  tirx.PrimFunc(params, body, buffer_map=...)
```

前者是前端常用路径，后者适合你做最小实验。

## 4. PrimExpr：值表达式

`PrimExpr` 是 TIRX 中所有标量/向量表达式的基类。定义主要在
`include/tvm/tirx/expr.h` 和 `include/tvm/tirx/var.h`。

常见节点可以按类别记：

| 类别 | 节点 | 说明 |
| --- | --- | --- |
| 常量 | `IntImm`, `FloatImm`, `StringImm` | `StringImm` 主要用于 assert message。 |
| 变量 | `Var`, `SizeVar` | 变量身份按对象区分，同名不代表同一个变量。 |
| 算术 | `Add`, `Sub`, `Mul`, `Div`, `Mod`, `FloorDiv`, `FloorMod`, `Min`, `Max` | 二元表达式，字段通常是 `a`, `b`。 |
| 比较 | `EQ`, `NE`, `LT`, `LE`, `GT`, `GE` | 结果是 bool dtype。 |
| 逻辑 | `And`, `Or`, `Not` | 条件表达式。 |
| 选择 | `Select` | 表达式级 if。 |
| 类型转换 | `Cast` | 字段是 `value`，dtype 存在 node dtype 中。 |
| Buffer 读 | `BufferLoad` | `buffer`, `indices`, `predicate`。 |
| 向量 | `Ramp`, `Broadcast`, `Shuffle` | vector index/value 构造。 |
| 调用 | `Call` | `op`, `args`, `dtype`, `annotations`。 |
| let/reduce | `Let`, `CommReducer`, `Reduce` | 局部绑定和规约表达式。 |

几个容易踩坑的点：

1. `BufferLoad` 是表达式，`BufferStore` 是语句。
2. `Call` 本身只是表达式；如果要把有副作用的 call 放进语句位置，需要
   `Evaluate(Call(...))`。
3. `Var` 的名字主要是打印和调试用途。分析时要看对象身份，不要只看
   `name_hint`。

表达式的遍历和改写在：

- C++: `include/tvm/tirx/expr_functor.h`, `src/tirx/ir/expr_functor.cc`
- Python: `python/tvm/tirx/expr_functor.py`

后面读 pass 时，经常会看到 `ExprVisitor`、`ExprMutator` 或 stmt mutator
里调用 `VisitExpr`。

## 5. Stmt：控制流和副作用

`Stmt` 定义在 `include/tvm/tirx/stmt.h`。这是 `PrimFunc.body` 的主体。

可以先把语句节点分成六组。

### 5.1 顺序和普通控制流

| 节点 | 关键字段 | 作用 |
| --- | --- | --- |
| `SeqStmt` | `seq` | 多条语句顺序执行。`SeqStmt::Flatten` 会消除 `Evaluate(0)` 并展开嵌套 seq。 |
| `For` | `loop_var`, `min`, `extent`, `kind`, `body`, `thread_binding`, `annotations`, `step` | for loop。 |
| `While` | `condition`, `body` | while loop。 |
| `IfThenElse` | `condition`, `then_case`, `else_case` | 语句级分支。 |
| `Break`, `Continue` | 无核心字段 | loop control。 |
| `Evaluate` | `value` | 把表达式作为语句执行，常用于副作用 call 或 no-op。 |

`ForKind` 有五种：

```text
kSerial
kParallel
kVectorized
kUnrolled
kThreadBinding
```

这里的 `kThreadBinding` 和 TIRX 的 `ExecScopeStmt` 是两个不同层次。前者仍是
loop 的属性，后者是显式硬件执行层级。

### 5.2 绑定和属性

| 节点 | 关键字段 | 作用 |
| --- | --- | --- |
| `Bind` | `var`, `value` | 在当前 enclosing scope 中绑定变量。这个节点没有 body。 |
| `AttrStmt` | `node`, `attr_key`, `value`, `body` | 给一段 body 附加属性。 |
| `AssertStmt` | `condition`, `error_kind`, `message_parts` | 运行时检查。 |

`Bind` 的设计值得特别注意。它没有 body，绑定对同一 enclosing scope 的后续
语句可见，这让 TIRX 可以表达更平坦的 sequence，而不是所有 binding 都嵌套成
`LetStmt(body)`。

### 5.3 Buffer 相关语句

| 节点 | 关键字段 | 作用 |
| --- | --- | --- |
| `BufferStore` | `buffer`, `value`, `indices`, `predicate` | 写 buffer。 |
| `DeclBuffer` | `buffer` | 声明一个 buffer。 |
| `AllocBuffer` | `buffer`, `annotations` | 分配 buffer。 |
| `BufferRegion` | `buffer`, `region` | 描述 buffer 区域，常用于 reads/writes。 |
| `MatchBufferRegion` | `buffer`, `source` | 表示 source region 可以 remap 到 target buffer。 |

前端里：

```python
B[i] = A[i] + 1
```

会落到：

```text
BufferStore(
  buffer=B,
  indices=[i],
  value=Add(BufferLoad(A, [i]), FloatImm(1))
)
```

而：

```python
A = T.match_buffer(a, (m, n), "float32")
```

如果 `a` 是函数参数，会进入 `PrimFunc.buffer_map`；如果参数是 block 内的
`BufferRegion`，会进入 `SBlock.match_buffers`。

### 5.4 SBlock 和 SBlockRealize

`SBlock` 是 schedule/block 语义的核心节点：

| 字段 | 作用 |
| --- | --- |
| `iter_vars` | block 轴，元素是 `IterVar`。 |
| `reads`, `writes` | 显式读写区域。 |
| `name_hint` | block 名字。 |
| `alloc_buffers` | block 内分配的 buffer。 |
| `match_buffers` | block 内 match buffer region。 |
| `annotations` | block 级 annotation。 |
| `init` | reduction block 的初始化语句。 |
| `body` | block 主体。 |

`SBlockRealize` 则负责把 block 的 `iter_vars` 绑定到具体的
`iter_values`，并附加 `predicate`。

前端中：

```python
with T.sblock("update"):
    vi, vj = T.axis.remap("SS", [i, j])
    T.reads(A[vi, vj])
    T.writes(B[vi, vj])
    B[vi, vj] = A[vi, vj]
```

IR 形状大致是：

```text
SBlockRealize(
  iter_values=[i, j],
  predicate=true,
  block=SBlock(
    name_hint="update",
    iter_vars=[IterVar(vi, data_par), IterVar(vj, data_par)],
    reads=[A region],
    writes=[B region],
    body=BufferStore(...)
  )
)
```

这里 `T.axis.remap` 不只是返回变量，它还会修改当前 `SBlockFrame` 的
`iter_vars` 和 `iter_values`。对应 C++ 实现在
`src/tirx/script/builder/ir.cc` 的 `axis::Remap`。

### 5.5 ExecScopeStmt：TIRX 的硬件执行层级

`ExecScopeStmt` 是 TIRX 区别于经典 TIR 的关键结构之一。定义在
`include/tvm/tirx/stmt.h`：

```cpp
class ExecScopeStmtNode : public StmtNode {
 public:
  ExecScope exec_scope;
  Stmt body;
};
```

`ExecScope` 定义在 `include/tvm/tirx/exec_scope.h`，核心字段是：

```cpp
class ExecScopeNode : public ffi::Object {
 public:
  ffi::Array<ScopeIdDef> scope_id_def;
  ScopeKind kind;
};
```

`ScopeKind` 从粗到细：

```text
world
kernel
cluster
cta
warpgroup
warp
thread
```

`ScopeIdDef` 则描述父 scope 到子 scope 的 id 绑定。它的关键字段：

| 字段 | 作用 |
| --- | --- |
| `def_ids` | 定义出的 id 变量。 |
| `extents` | id 的 extent。`None` 表示 deferred，后续在 `LowerTIRx` 入口推断。 |
| `scope` | `ScopeBinding`，例如 `kCtaThread`、`kWarpThread`。 |
| `preferred_extents` | cluster 到 cta 这类场景的 preferred launch extent。 |

这个设计把 GPU/加速器执行层级作为 IR 的一等结构，而不是只藏在
`threadIdx.x` 这类后端 intrinsic 里。这样 pass 可以在 lowering 前分析
kernel/cta/warp/thread 的层级关系。

### 5.6 TilePrimitiveCall

`TilePrimitiveCall` 定义在 `include/tvm/tirx/tirx_stmt.h`：

```cpp
class TilePrimitiveCallNode : public StmtNode {
 public:
  tvm::Op op;
  ffi::Array<ffi::Any> args;
  ffi::Map<ffi::String, Buffer> workspace;
  ffi::Map<ffi::String, ffi::Any> config;
  ffi::Optional<ffi::String> dispatch;
};
```

它表示高层 tile primitive 调用，不是最终 codegen 形态。可以把它理解成：

```text
先保留高层 tile 操作语义
  op + args + workspace + config + dispatch
后续 pass 根据 target/layout/scope 选择实现并展开
```

读这部分时建议从：

- `python/tvm/tirx/operator/tile_primitive`
- `src/tirx/transform/tile_primitive_dispatch.cc`
- `src/tirx/transform/lower_tirx.cc`

开始。

## 6. Buffer：内存对象不是简单指针

`Buffer` 定义在 `include/tvm/tirx/buffer.h`。它是 TIRX 表达内存访问的核心：

| 字段 | 作用 |
| --- | --- |
| `data` | 底层指针变量，类型通常是 pointer/handle。 |
| `dtype` | 元素类型。 |
| `shape` | 逻辑访问维度。 |
| `strides` | 显式 stride，空数组表示 contiguous。 |
| `elem_offset` | 以 element 为单位的偏移。 |
| `axis_separators` | flatten 多维轴时的分隔信息。 |
| `name` | 打印和调试名。 |
| `data_alignment`, `offset_factor` | 对齐和 offset 约束。 |
| `buffer_type` | `kDefault` 或 `kAutoBroadcast`。 |
| `layout` | TIRX 扩展，buffer 的逻辑到物理/layout 映射。 |
| `allocated_addr` | 多维 allocation 地址，例如某些硬件的 bank/offset。 |

为什么 buffer 要这么复杂？因为在底层 IR 中，buffer 既要服务代码生成，又要服务
分析和约束检查：

- shape/stride/offset 决定 index 如何 flatten。
- `buffer_map` 让函数参数携带结构化约束。
- `layout` 让 tile/swizzle/thread/memory 映射可以在 IR 中显式存在。
- `allocated_addr` 给非普通线性内存提供表达空间。

一个很好的阅读切入点是 `BufferNode::ElemOffset` 和 Python 的
`Buffer.__getitem__`/`__setitem__` 路径。前者解释 index 到 offset 的 IR 计算，
后者解释 Python 访问如何变成 `BufferLoad` 或 `BufferRegion`。

## 7. Layout：TIRX 把 layout 做成一等 IR

`Layout` 定义在 `include/tvm/tirx/layout.h`。它不是注释，而是有虚函数接口的
IR object：

```cpp
class LayoutNode : public ffi::Object {
 public:
  virtual bool CompatibleWithShape(...) const = 0;
  virtual bool VerifyWellFormed() const = 0;
  virtual PrimExpr GetSize(...) const = 0;
  virtual PrimExpr GetSpan(...) const = 0;
  virtual ffi::Map<ffi::String, PrimExpr> Apply(...) const = 0;
  virtual Layout Canonicalize() const = 0;
  virtual Layout Tile(...) const = 0;
  virtual ffi::Optional<Layout> Slice(...) const = 0;
  virtual Layout DirectSum(...) const = 0;
};
```

主要实现类：

| 节点 | 字段 | 作用 |
| --- | --- | --- |
| `Axis` | `name`，以及 registry attrs | 表示布局轴，可带 scope/subscope/fuser/splitter。 |
| `Iter` | `extent`, `stride`, `axis` | layout 中某个 axis 的一段迭代。 |
| `TileLayout` | `shard`, `replica`, `offset` | tile 映射的核心结构。 |
| `SwizzleLayout` | `per_element`, `swizzle_len`, `atom_len`, `swizzle_inner` | 表示 swizzle。 |
| `ComposeLayout` | `swizzle`, `tile_layout` | swizzle 和 tile layout 的组合。 |

直观理解：

```text
logical index
  -> layout.Apply(...)
  -> {axis_name: physical/thread/memory coordinate}
```

传统 TIR 里很多 layout 语义会散落在 index 算式、buffer flatten 或 schedule
约定里。TIRX 把 layout 作为对象挂到 `Buffer` 上，让 tile primitive 和 lowering
可以直接读出“这个 buffer 是如何映射到 memory/thread axes 的”。

## 8. Python wrapper 和 C++ node 是同一个 IR 的两面

TIRX IR 节点的权威定义在 C++ 头文件，Python 侧通过 FFI 注册到同一个对象系统。

例如 `tirx.For`：

- C++ node: `include/tvm/tirx/stmt.h`
- C++ constructor: `src/tirx/ir/stmt.cc`
- Python wrapper: `python/tvm/tirx/stmt.py`

Python 侧的类通常只是：

```python
@tvm_ffi.register_object("tirx.For")
class For(Stmt):
    def __init__(...):
        self.__init_handle_by_constructor__(_ffi_api.For, ...)
```

所以学习时建议采用这个顺序：

```text
1. 先读 include/tvm/tirx/*.h
   搞清楚字段和不变量

2. 再读 src/tirx/ir/*.cc
   看 constructor 如何校验、推导 dtype、注册 reflection

3. 最后读 python/tvm/tirx/*.py
   看 Python 侧做了哪些兼容、转换和语法糖
```

不要反过来只看 Python wrapper。那样很容易错过 C++ node 的真实字段和 pass
实际依赖的结构。

## 9. IR 是如何从 TVMScript builder 生成的

你已经学过前端的话，这里只需要记一个机制：TIRX parser 不直接 new 大多数 IR
节点，而是调用 builder API。builder API 操作 C++ `IRBuilder` 的 frame 栈。

核心文件：

- `python/tvm/tirx/script/parser/parser.py`
- `python/tvm/tirx/script/builder/ir.py`
- `src/tirx/script/builder/ir.cc`
- `src/tirx/script/builder/frame.cc`
- `src/tirx/script/builder/utils.h`

以 `for i in T.grid(128)` 为例：

```text
parser visit doc.For
  eval_expr(T.grid(128))
    -> Python builder 调 C++ Grid(...)
    -> 创建 ForFrame(vars, doms, f_make_for_loop)

  with ForFrame:
    frame 入 IRBuilder.current().frames
    parse loop body

  退出 ForFrame:
    frame.cc: ForFrameNode::ExitWithScope()
    -> f_make_for_loop(vars, doms, steps, AsStmt(stmts))
    -> AddToParent(For(...))
```

`AddToParent` 在 `src/tirx/script/builder/utils.h`：

```cpp
if (builder->frames.empty()) {
  builder->result = stmt;
} else if (top is TIRFrame) {
  top->stmts.push_back(stmt);
}
```

也就是说：

```text
进入 frame 时建立父子上下文
普通 stmt 加到当前最内层 frame
退出 frame 时把子 stmt 折叠成一个父 stmt
```

这就是你在 parser Python 文件里看不到太多 `IRBuilder.current()` 的原因。
`IRBuilder.current()` 被封装在 frame enter/exit 和 C++ builder API 内部。

## 10. IR 遍历与改写：读 pass 的入口

掌握节点后，下一步要看 pass。TIRX pass 基本都建立在 visitor/mutator 上。

C++ 侧：

- `include/tvm/tirx/expr_functor.h`
- `include/tvm/tirx/stmt_functor.h`
- `src/tirx/ir/expr_functor.cc`
- `src/tirx/ir/stmt_functor.cc`

Python 侧：

- `python/tvm/tirx/expr_functor.py`
- `python/tvm/tirx/stmt_functor.py`

`StmtFunctor` 对每种 stmt 做 vtable dispatch：

```cpp
VisitStmt_(const ForNode* op)
VisitStmt_(const BufferStoreNode* op)
VisitStmt_(const SBlockNode* op)
VisitStmt_(const ExecScopeStmtNode* op)
VisitStmt_(const TilePrimitiveCallNode* op)
```

`StmtVisitor` 用于只读遍历，`StmtMutator` 用于返回改写后的节点。读 transform
源码时，基本就是找它 override 了哪些节点：

```text
override VisitStmt_(ForNode)
  -> 改 loop

override VisitStmt_(BufferStoreNode)
  -> 改内存访问或 dtype

override VisitStmt_(ExecScopeStmtNode)
  -> 改执行层级

override VisitStmt_(TilePrimitiveCallNode)
  -> 展开 tile primitive
```

尤其要注意 buffer 的访问策略。`StmtVisitor` 里区分：

```text
VisitBufferDef
  访问 buffer 定义点，如 AllocBuffer、DeclBuffer、SBlock alloc_buffers

VisitBufferUse
  访问 buffer 使用点，如 BufferStore、BufferLoad、SBlock reads/writes
```

这个区分是为了避免每次使用 buffer 都重复访问 shape/stride/elem_offset 等定义信息。

## 11. 从 IR 到 lowering pipeline

IR 节点不是孤立存在的。`tvm.tirx.build` 会把 `PrimFunc/IRModule` 放进一串 pass。

入口：

- `python/tvm/tirx/build.py`
- `python/tvm/tirx/compilation_pipeline.py`

`build` 的核心流程：

```text
PrimFunc -> IRModule
BindTarget(target)
pipeline(mod)
split_host_device_mods(mod)
finalize host/device
target.build.*
runtime module
```

`pipeline="tirx"` 时，主要 pass 顺序在 `tirx_pipeline()`：

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

学习时不建议一开始逐个 pass 深挖。先抓住几条主线：

| 主线 | 相关 pass | 关注点 |
| --- | --- | --- |
| TIRX 专有语义降低 | `LowerTIRx`, `LowerTIRxOpaque` | `ExecScopeStmt`、scope id、tile primitive、opaque 结构如何消失或变低层。 |
| 内存形态规范化 | `FlattenBuffer`, `VerifyMemory`, `LowerWarpMemory` | 多维 buffer、layout、storage scope 如何变成 codegen 更容易处理的形态。 |
| 常规优化 | `Simplify`, `CommonSubexprElim`, `VectorizeLoop`, `UnrollLoop` | 表达式和 loop 结构如何优化。 |
| host/device 拆分 | `AnnotateDeviceRegions`, `SplitHostDevice`, `MakePackedAPI`, `LowerDeviceKernelLaunch` | kernel 函数和 host launch wrapper 如何分离。 |
| dtype 合法化 | `BF16*`, `FP8*`, `NarrowDataType`, `LowerCustomDatatypes` | 特殊 dtype 如何在目标上落地。 |

## 12. 建议的学习路径

如果你的目标是继续系统学习 TIRX IR，我建议按下面顺序走。

### 第一步：PrimFunc + Stmt 树

读：

- `include/tvm/tirx/function.h`
- `include/tvm/tirx/stmt.h`
- `python/tvm/tirx/function.py`
- `python/tvm/tirx/stmt.py`

要回答：

- `PrimFunc.params` 和 `buffer_map` 为什么要分开？
- `BufferLoad` 和 `BufferStore` 为什么一个是 expr，一个是 stmt？
- `SeqStmt::Flatten` 为什么要消除 `Evaluate(0)`？
- `Bind` 没有 body，这和传统 let 有什么区别？

### 第二步：表达式和 buffer

读：

- `include/tvm/tirx/expr.h`
- `include/tvm/tirx/var.h`
- `include/tvm/tirx/buffer.h`
- `src/tirx/ir/buffer.cc`

要回答：

- `Var` 的 identity 和 name_hint 各自有什么用？
- `Buffer.shape/strides/elem_offset/axis_separators` 如何共同决定 offset？
- 为什么 `Buffer` 上需要 `layout`？

### 第三步：SBlock 与 schedule 语义

读：

- `include/tvm/tirx/stmt.h` 的 `SBlock/SBlockRealize`
- `src/tirx/script/builder/ir.cc` 的 `axis::Remap`, `Reads`, `Writes`
- `src/tirx/script/builder/frame.cc` 的 `SBlockFrameNode::ExitWithScope`

要回答：

- `SBlock.iter_vars` 和 `SBlockRealize.iter_values` 如何配对？
- `T.axis.remap` 是如何找到 loop domain 的？
- `reads/writes` 缺失时为什么会有 `tirx.script_parsing_detect_access` annotation？

### 第四步：ExecScope

读：

- `include/tvm/tirx/exec_scope.h`
- `include/tvm/tirx/stmt.h` 的 `ExecScopeStmt`
- `src/tirx/script/builder/ir.cc` 的 `Kernel/CTA/Warp/Thread/ScopeId`
- `src/tirx/transform/lower_tirx.cc`

要回答：

- `ScopeKind` 和 `ScopeBinding` 分别表达什么？
- deferred extent 是怎么设计的？
- `with T.thread()` 最终如何变成目标后端 thread 语义？

### 第五步：Layout 和 tile primitive

读：

- `include/tvm/tirx/layout.h`
- `src/tirx/ir/layout/*.cc`
- `include/tvm/tirx/tirx_stmt.h`
- `python/tvm/tirx/operator/tile_primitive`
- `src/tirx/transform/tile_primitive_dispatch.cc`

要回答：

- `TileLayout.shard/replica/offset` 分别表示什么？
- `Layout.Apply` 返回的 map 为什么是 `axis_name -> PrimExpr`？
- `TilePrimitiveCall` 为什么不直接 lowering 成普通 stmt？

### 第六步：pass 和 build

读：

- `python/tvm/tirx/compilation_pipeline.py`
- `python/tvm/tirx/build.py`
- `src/tirx/transform/*.cc`

先挑三类 pass：

1. `LowerTIRx`
2. `FlattenBuffer`
3. `SplitHostDevice` / `MakePackedAPI` / `LowerDeviceKernelLaunch`

把这三类看懂后，再回头看 dtype legalize、vectorize、unroll、CSE 会容易很多。

## 13. 一个实用的源码阅读方法

每遇到一个 IR 节点，按四步走：

```text
1. 头文件
   include/tvm/tirx/xxx.h
   看字段、注释、不变量

2. C++ 构造
   src/tirx/ir/xxx.cc
   看 constructor 做了什么校验和推导

3. Python wrapper
   python/tvm/tirx/xxx.py
   看 Python 侧有哪些语法糖和兼容逻辑

4. pass 使用
   rg "NodeName" src/tirx/transform python/tvm/tirx
   看谁真正消费这个节点
```

例如研究 `ExecScopeStmt`：

```text
include/tvm/tirx/stmt.h
  字段 exec_scope/body

include/tvm/tirx/exec_scope.h
  ScopeKind/ScopeBinding/ScopeIdDef

src/tirx/script/builder/frame.cc
  ExecScopeFrameNode::ExitWithScope 如何构造 ExecScopeStmt

src/tirx/transform/lower_tirx.cc
  lowering 如何消费 ExecScopeStmt
```

研究 `BufferStore`：

```text
include/tvm/tirx/stmt.h
  buffer/value/indices/predicate

src/tirx/ir/stmt.cc
  constructor 和 dtype/结构校验

src/tirx/script/builder/ir.cc
  T.buffer_store 如何做 dtype cast 并 AddToParent

src/tirx/transform/flatten_buffer.cc
  多维 buffer store 如何 flatten
```

## 14. 总结

从 TVMScript 过渡到 TIRX IR，最重要的是把“写法”与“数据结构”分开：

```text
TVMScript:
  给人写的 Python-like DSL

TIRX builder:
  把 DSL 解释成 IRBuilder frame 和 stmt

TIRX IR:
  给 pass/lowering/codegen 消费的数据结构
```

TIRX IR 的学习主线可以概括成：

```text
PrimFunc 是函数单位
Stmt 是控制流和副作用
PrimExpr 是值表达式
Buffer 是结构化内存对象
SBlock 是 schedule/block 单元
ExecScopeStmt 是硬件执行层级
Layout 是一等映射对象
TilePrimitiveCall 是高层 tile 操作的过渡节点
```

掌握这张图之后，再看 `LowerTIRx`、`FlattenBuffer`、`SplitHostDevice` 这些 pass，
就不会觉得它们是在随意改树，而是在逐步消解 TIRX IR 中的高层结构，把它们变成
目标后端能直接 codegen 的低层形态。
