"""Runnable walkthrough for the TIRX codegen example in codegen.md.

Do not hand-write the lowered IR from ``lowered.script(show_meta=True)`` back
into a @T.prim_func.  Some printed nodes, such as T.large_uint_imm, are meant as
printer output for an existing IR object, while the normal parser-side way to
construct this constant is tvm.tirx.const(..., "uint64").
"""


import tvm
from tvm.script import ir as I
from tvm.script import tirx as T

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

def s_tir_pipeline_before_split_host_device():
    """Return the default s_tir pipeline prefix just before SplitHostDevice.

    This mirrors 3rdparty/tvm/python/tvm/s_tir/pipeline.py up to and including
    AnnotateDeviceRegions, and deliberately stops before SplitHostDevice.
    """

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        pass_ctx = tvm.transform.PassContext.current()
        config = pass_ctx.config
        passes = [
            tvm.s_tir.transform.CanonicalizeLoop(),
            tvm.s_tir.transform.LowerCrossThreadReduction(),
            tvm.s_tir.transform.LowerInitBlock(),
            tvm.s_tir.transform.PlanAndUpdateBufferAllocationLocation(),
            tvm.s_tir.transform.ConvertBlocksToOpaque(),
            tvm.s_tir.transform.LiftThreadBinding(),
            tvm.s_tir.transform.ManifestSharedMemoryLocalStage(),
            tvm.s_tir.transform.CompactBufferAllocation(),
            tvm.s_tir.transform.LowerAutoCopy(),
            tvm.s_tir.transform.UnifyThreadBinding(),
            tvm.s_tir.transform.LowerMatchBuffer(),
            tvm.tirx.transform.Simplify(),
            tvm.s_tir.transform.InjectPermutedLayout(),
            tvm.s_tir.transform.AnnotateIrregularLoop(),
            tvm.s_tir.transform.InjectSoftwarePipeline(),
            tvm.s_tir.transform.TransformMmaBufferLayout(),
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.tirx.transform.FlattenBuffer(),
            tvm.tirx.transform.BF16ComputeLegalize(),
            tvm.tirx.transform.NarrowDataType(32),
            tvm.s_tir.transform.LoopPartition(),
            tvm.tirx.transform.VectorizeLoop(
                not bool(config.get("tirx.disable_vectorize", False))
            ),
            tvm.s_tir.transform.InjectVirtualThread(),
            tvm.s_tir.transform.InjectDoubleBuffer(),
        ]
        if not bool(config.get("tirx.disable_storage_rewrite", False)):
            passes.append(tvm.tirx.transform.StorageRewrite())
        if config.get("tirx.use_async_copy", False):
            passes.append(tvm.s_tir.transform.LowerAsyncDMA())
        passes.extend(
            [
                tvm.s_tir.transform.HoistIfThenElse(),
                tvm.tirx.transform.UnrollLoop(),
                tvm.s_tir.transform.RenormalizeSplitPattern(),
                tvm.tirx.transform.Simplify(),
                tvm.tirx.transform.RemoveNoOp(),
                tvm.s_tir.transform.RewriteUnsafeSelect(),
            ]
        )
        if bool(config.get("tirx.instrument_bound_checkers", False)):
            passes.append(tvm.s_tir.transform.InstrumentBoundCheckers())
        if bool(config.get("tirx.ptx_ldg32", False)):
            passes.append(tvm.s_tir.transform.InjectPTXLDG32(True))
        if not bool(config.get("tirx.disable_cse_tir", False)):
            passes.append(tvm.tirx.transform.CommonSubexprElim())
        if bool(config.get("tirx.instrument_lwp", False)):
            passes.append(tvm.s_tir.transform.InstrumentProfileIntrinsics())
        passes.extend(
            [
                tvm.tirx.transform.FP8ComputeLegalize(),
                tvm.s_tir.transform.VerifyVTCMLimit(),
                tvm.s_tir.transform.LowerVtcmAlloc(),
                tvm.tirx.transform.VerifyMemory(),
                tvm.tirx.transform.AnnotateEntryFunc(),
                tvm.s_tir.transform.ThreadSync("shared"),
                tvm.s_tir.transform.ThreadSync("shared.dyn"),
                tvm.s_tir.transform.ThreadSync("warp"),
                tvm.s_tir.transform.InferFragment(),
                tvm.s_tir.transform.LowerThreadAllreduce(),
            ]
        )
        if bool(config.get("tirx.use_async_copy", False)):
            passes.append(tvm.s_tir.transform.InjectPTXAsyncCopy())
        if bool(config.get("tirx.ptx_ldg32", False)):
            passes.append(tvm.s_tir.transform.InjectPTXLDG32())
        passes.append(tvm.tirx.transform.AnnotateDeviceRegions())
        return tvm.ir.transform.Sequential(passes)(mod)

    return _pipeline


def s_tir_pipeline_after_split_host_device():
    """Return the default s_tir pipeline suffix after SplitHostDevice."""

    return tvm.ir.transform.Sequential(
        [
            tvm.s_tir.transform.MergeSharedMemoryAllocations(),
            tvm.tirx.transform.MakePackedAPI(),
            tvm.tirx.transform.FP8StorageLegalize(),
            tvm.tirx.transform.BF16StorageLegalize(),
            tvm.tirx.transform.LowerDeviceKernelLaunch(),
        ]
    )


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def inspect_module_source(rt_mod, preferred_formats):
    """Return the first inspectable source for a runtime module."""
    for fmt in preferred_formats:
        try:
            return fmt, rt_mod.inspect_source(fmt)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    try:
        return "", rt_mod.inspect_source()
    except Exception:  # pylint: disable=broad-exception-caught
        return "", ""


def main() -> None:
    target = tvm.target.Target("cuda")
    target_host = tvm.target.Target("llvm" if tvm.runtime.enabled("llvm") else "c")

    print_section("0. Original Module")
    print(Module.script())

    print_section("1. After BindTarget")
    bound = tvm.tirx.transform.BindTarget(target.with_host(target_host))(Module)
    print(bound.script())

    print_section("2. After s_tir pipeline passes, before SplitHostDevice")
    before_split = s_tir_pipeline_before_split_host_device()(bound)
    print(before_split.script(show_meta=True))

    print_section("3. PrimFunc before SplitHostDevice")
    for gvar, func in before_split.functions.items():
        print(f"\n--- {gvar.name_hint} ---")
        print(func.script(show_meta=True))

    print_section("4. After SplitHostDevice")
    after_split = tvm.tirx.transform.SplitHostDevice()(before_split)
    print(after_split.script(show_meta=True))

    print_section("5. PrimFuncs after SplitHostDevice")
    for gvar, func in after_split.functions.items():
        print(f"\n--- {gvar.name_hint} ---")
        print(func.script(show_meta=True))

    print_section("6. After remaining s_tir pipeline passes")
    ready_for_codegen = s_tir_pipeline_after_split_host_device()(after_split)
    print(ready_for_codegen.script(show_meta=True))

    print_section("7. Split host/device modules for codegen")
    from tvm.tirx.build import codegen_build, split_host_device_mods

    host_mod, device_mod_dict = split_host_device_mods(ready_for_codegen)
    print("\n--- host_mod before finalization ---")
    print(host_mod.script(show_meta=True))
    for device_target, device_mod in device_mod_dict.items():
        print(f"\n--- device_mod before finalization: {device_target} ---")
        print(device_mod.script(show_meta=True))

    print_section("8. Finalize device modules and run target.build.<device>")
    _, finalize_host_passes, finalize_device_passes = tvm.tirx.get_tir_pipeline("default")
    device_runtime_modules = []
    for device_target, device_mod in device_mod_dict.items():
        finalized_device_mod = finalize_device_passes()(device_mod)
        print(f"\n--- finalized device_mod: {device_target} ---")
        print(finalized_device_mod.script(show_meta=True))

        print(f"\n--- target.build.{device_target.kind.name} result ---")
        device_runtime_mod = codegen_build(finalized_device_mod, device_target)
        device_runtime_modules.append(device_runtime_mod)
        fmt, source = inspect_module_source(device_runtime_mod, ["cuda", "cu", "ptx"])
        if source:
            suffix = f" ({fmt})" if fmt else ""
            print(f"device source{suffix}:")
            print(source)
        else:
            print("device runtime module has no inspectable source")

    print_section("9. Finalize host module and run target.build.<host>")
    finalized_host_mod = finalize_host_passes()(host_mod)
    print("\n--- finalized host_mod ---")
    print(finalized_host_mod.script(show_meta=True))

    print(f"\n--- target.build.{target_host.kind.name} result ---")
    host_runtime_mod = codegen_build(finalized_host_mod, target_host)
    host_formats = ["ll", "llvm"] if target_host.kind.name == "llvm" else ["c", "cc", "cpp"]
    fmt, source = inspect_module_source(host_runtime_mod, host_formats)
    if source:
        suffix = f" ({fmt})" if fmt else ""
        print(f"host source/IR{suffix}:")
        print(source)
    else:
        print("host runtime module has no inspectable source")

    print_section("10. Import device modules into host runtime module")
    for device_runtime_mod in device_runtime_modules:
        host_runtime_mod.import_module(device_runtime_mod)
    print("host runtime imports:", len(host_runtime_mod.imports))


if __name__ == "__main__":
    main()
