# 第 1 周：TVM IR 基础

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
- `Target` 如何表示 `cuda -host=llvm` 这类组合 target。
- `Pass`、`Sequential`、`PassContext` 的调用模型是什么。

建议实验：

1. 写一个最小 `IRModule`，打印 `mod.script()`。
2. 查一个 pass 的 Python wrapper 和 C++ 注册函数如何对应。
3. 在 Python 中构造 `tvm.target.Target("cuda")`，观察 `kind`、`keys`、`host`。

阶段产出：

- `IRModule / PrimFunc / Target / Pass` 的关系图。
- 一份 “Python API 调用如何进入 C++ FFI” 的短笔记。

# TVM 的 `ObjectRef` / `Object` / `ObjectPtr`？
Object      = 带 TVMFFIObject header 的堆对象基类
ObjectPtr   = intrusive ref-counted owning pointer
ObjectRef   = 对外的 typed handle，内部持有 ObjectPtr<Object>

# FFI 注册模型 是什么？
FFI 注册系统可以拆成四张表/映射来看：
1. TypeTable
    type_key / type_index / 父子关系 / fields / methods / attrs
2. GlobalFunctionTable
    string name -> ffi::Function
3. Python object registry
    type_index / type_key -> Python class
4. TypeAttr table
    type_index + attr_name -> Any
## 1.对象类型注册 TypeTable
对象类型通常这样声明：
``` c++
class IRModuleNode : public ffi::Object {
TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.IRModule", IRModuleNode, ffi::Object);
};
```

这个宏给 IRModuleNode 定义 _type_key、RuntimeTypeIndex()、_GetOrAllocRuntimeTypeIndex()。后者会调用：

``` c++
TVMFFITypeGetOrAllocIndex(...)
```
最终落到 TypeTable::GetOrAllocTypeIndex，见 3rdparty/tvm/3rdparty/tvm-ffi/src/ffi/object.cc:81。
TypeTable 里保存：

type_key -> type_index
type_index -> TypeInfo
type_ancestors
fields
methods
metadata
type attrs

它也负责父子类型关系和快速 IsInstance 检查所需的 child slot 分配。

## 2.反射注册
仅声明type key 不等于完整注册。字段、方法、构造器靠ObjectDef：
refl::ObjectDef<IRModuleNode>()
      .def_ro("functions", &IRModuleNode::functions)
      .def_ro("source_map", &IRModuleNode::source_map)
      .def_ro("attrs", &IRModuleNode::attrs);


  例子在 3rdparty/tvm/include/tvm/ir/module.h:135。


  ObjectDef<T>() 构造时会拿到 type index；.def_ro() 最终调用 TVMFFITypeRegisterField；.def() 调用 TVMFFITypeRegisterMethod；析构时还会注册 __ffi_new__、__ffi_init__、__ffi_shallow_copy__ 等属性。实现集中在 3rdparty/tvm/3rdparty/tvm-ffi/include/tvm/ffi/reflection/registry.h:693。

  所以：


  TVM_FFI_DECLARE_OBJECT_INFO
    让类型进入 type system


  refl::ObjectDef<T>()
    让字段、方法、构造、反射能力进入 FFI

## 3.全局函数注册 GlobalFunctionTable
全局函数用：
refl::GlobalDef()
    .def("ir.IRModule", ...)
    .def("ir.Module_Clone", ...);


例子在 3rdparty/tvm/src/ir/module.cc:231。


GlobalDef().def 会把 C++ lambda/function 包成 ffi::Function，然后调用：


TVMFFIFunctionSetGlobalFromMethodInfo(...)


运行时落到 GlobalFunctionTable::Update，表结构在 3rdparty/tvm/3rdparty/tvm-ffi/src/ffi/function.cc:51。这张表就是：


"ir.IRModule" -> ffi::Function
"relax.transform.FoldConstant" -> ffi::Function
...


Python 的 tvm_ffi.get_global_func("ir.IRModule") 最终调用 TVMFFIFunctionGetGlobal，拿回同一个 ffi::Function。
## 4.调用模型
ffi::Function 是一个 Object，调用时走：

Python/C++ caller
-> TVMFFIAny[] args
-> TVMFFIFunctionCall
-> FunctionObj::safe_call
-> C++ typed lambda
-> TVMFFIAny result


TVMFFIFunctionCall 在 3rdparty/tvm/3rdparty/tvm-ffi/src/ffi/function.cc:191。


类型转换靠 TypeTraits。比如 ObjectRef 会被转成：


TVMFFIAny {
type_index = object->type_index
v_obj = TVMFFIObject*
}


所以 FFI 调用不是只传 void*，而是每个值都带 type_index。

## 5. Python 包装注册
@tvm_ffi.register_object("ir.IRModule")
  class IRModule(Object):
      pass


会先用 type key 查 C++ type index，然后把：


type_index -> Python class
type_key -> TypeInfo


登记到 Python registry。入口在 3rdparty/tvm/3rdparty/tvm-ffi/python/tvm_ffi/registry.py:36。


当 C++ 返回一个 ObjectRef，Python 侧根据 result.type_index 找对应 Python class，构造 wrapper。逻辑在 3rdparty/tvm/3rdparty/tvm-ffi/python/tvm_ffi/cython/object.pxi:471。

# `IRModule` 如何保存 `GlobalVar -> BaseFunc`
- functions 是真正的 GlobalVar -> BaseFunc 表。
- global_var_map_ 是辅助索引，保存 name_hint -> GlobalVar，用于 mod["main"] 这种按名字查找，并保证模块内名字唯一。
- 构造 IRModule 时，会直接 n->functions = std::move(functions)，然后遍历 functions 填充 global_var_map_：见 3rdparty/tvm/src/ir/module.cc:42。
- 新增/更新函数走 IRModuleNode::AddUnchecked，先 functions.Set(var, func)，再更新 global_var_map_：见 3rdparty/tvm/src/ir/module.cc:153。
- 查找时，Lookup(GlobalVar) 直接查 functions.find(var)；Lookup(String) 先从 global_var_map_ 找到 canonical GlobalVar，再查 functions：见 3rdparty/tvm/src/ir/module.cc:182。

注意一个容易踩的点：GlobalVar 里只有 name_hint，但作为普通 map key 时不要依赖“同名就相等”。std::hash<tvm::GlobalVar> 和 std::equal_to<tvm::GlobalVar> 使用的是对象指针身份，见 3rdparty/tvm/include/tvm/ir/expr.h:789。所以：

auto gv = mod->GetGlobalVar("main");
BaseFunc f = mod->Lookup(gv);      // 正确

BaseFunc f2 = mod->Lookup("main"); // 也正确

BaseFunc bad = mod->Lookup(GlobalVar("main")); // 通常不对：这是另一个 GlobalVar 对象

Python 层只是包装这套 C++ API。IRModule({"main": func}) 会把字符串 key 转成 GlobalVar("main")，再通过 FFI 调 C++ 构造；mod["main"] 调 Module_Lookup_str，mod[gv] 调 Module_Lookup，见 3rdparty/tvm/python/tvm/ir/module.py:42。


# `Target` 如何用嵌套 `host` 字段表示 `cuda -host=llvm` 这类组合 target，而不是依赖旧式 CLI target string。

在这个 3rdparty/tvm 里，组合 target 的规范表达不是 "cuda -host=llvm"，而是 Target 对象里嵌套一个 host target。

用法：

import tvm

t = tvm.target.Target("cuda", host="llvm")
print(t.kind.name)        # cuda
print(t.host.kind.name)   # llvm
print(t.export())

等价的 dict 写法是：

t = tvm.target.Target({
    "kind": "cuda",
    "host": {"kind": "llvm"},
})

导出形态大致是：

{
    "kind": "cuda",
    "host": {
        "kind": "llvm",
        ...
    },
    ...
}


# `Pass`、`Sequential`、`PassContext` 的调用模型是什么
调用模型可以理解成：Pass 是统一可调用对象，Sequential 是一个特殊的 Pass，PassContext 是线程局部的运行环境。

  典型 Python 用法：

  seq = tvm.transform.Sequential([
      tvm.tirx.transform.Simplify(),
      tvm.tirx.transform.FlattenBuffer(),
  ])

  with tvm.transform.PassContext(
      opt_level=3,
      disabled_pass=["tirx.VectorizeLoop"],
      config={"tirx.disable_vectorize": True},
  ):
      mod = seq(mod)

  调用链大致是：

  Python pass(mod)
    -> FFI transform.RunPass
    -> C++ Pass::operator()(mod)
    -> PassContext::Current()
    -> Pass::operator()(mod, pass_ctx)
    -> InstrumentBeforePass
    -> 具体 PassNode::operator()(mod, pass_ctx)
    -> InstrumentAfterPass
    -> 返回新 IRModule

  关键源码：

  - Python Pass.__call__ 只是调 FFI：
    3rdparty/tvm/python/tvm/ir/transform.py:157

  - C++ Pass::operator()(mod) 会取当前 PassContext：
    3rdparty/tvm/src/ir/transform.cc:290

  - 真正执行前后会跑 pass instrument：
    3rdparty/tvm/src/ir/transform.cc:294

  PassContext 是 thread-local 栈。with PassContext(...) 进入时压栈，退出时弹栈；没有显式 context 时用默认 context：

  - thread-local default/context stack：
    3rdparty/tvm/src/ir/transform.cc:43

  - Current() 返回栈顶或默认 context：
    3rdparty/tvm/src/ir/transform.cc:77

  - Python __enter__ / __exit__ 对应 enter/exit：
    3rdparty/tvm/python/tvm/ir/transform.py:108

  Sequential 本身也是一个 Pass。它的 operator() 会按列表顺序执行子 pass，并在执行每个子 pass 前检查 PassContext.PassEnabled：

  for pass in passes:
      if not pass_ctx.PassEnabled(pass.info):
          continue
      for required_name in pass.info.required:
          mod = GetPass(required_name)(mod, pass_ctx)
      mod = pass(mod, pass_ctx)

  源码在 3rdparty/tvm/src/ir/transform.cc:470。

  启用规则是：

  disabled_pass 命中 -> 不跑
  required_pass 命中 -> 强制跑
  否则 opt_level >= pass.info.opt_level 才跑

  源码在 3rdparty/tvm/src/ir/transform.cc:94。

  注意一个细节：opt_level / disabled_pass 这套过滤主要在 SequentialNode::operator() 里做。直接调用单个 pass(mod) 时，会走当前 PassContext 和 instrument，但不会由 Sequential 帮你按 opt_level 跳过它。

  具体 pass 类型上：

  - ModulePass：IRModule -> IRModule，直接调用 pass_func(mod, ctx)，见 3rdparty/tvm/src/ir/transform.cc:395。
  - tirx.PrimFuncPass：外部看仍是 Pass，但内部遍历 IRModule.functions，只对 tirx.PrimFunc 调 pass_func(func, mod, ctx)，见 3rdparty/tvm/src/tirx/ir/transform.cc:109。
  - Sequential：组合多个 pass，常用于 pipeline，例如 tirx pipeline 最后就是 tvm.ir.transform.Sequential(passes)(mod)，见 3rdparty/tvm/python/tvm/tirx/compilation_pipeline.py:100。