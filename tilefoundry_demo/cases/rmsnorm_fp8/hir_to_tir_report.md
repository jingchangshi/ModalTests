# HIR → TIR report: `rms_norm_quant_opt.py` (RMSNorm + FP8)

**Verdict: the conversion succeeded — but not from the file as authored.**
`schedule finalize` refuses `rms_norm_quant_opt.py` for two independent, structural reasons.
A semantically identical, explicitly scheduled HIR was written beside it and lowered to real,
verified TIR. Nothing was hand-written into the `.tir`.

| | |
|---|---|
| Target file (authored HIR) | `rms_norm_quant_opt.py` — **unchanged**, still fails finalize |
| Scheduled HIR (new) | `rms_norm_quant_opt_scheduled.py` |
| Emitted TIR | `rms_norm_quant_opt.tir` (185 lines, sha256 `7dd70481…b3a4`) |
| TileFoundry | `0.1.dev203+g1393b1945`, commit `1393b1945629adf2a3a8ec5747499d7a63d35f5a` |
| Backend | `nvidia.h200_sxm` static description — **no GPU, no driver, no Modal** |

---

## 1. How TileFoundry is loaded here

Two separate things exist on this machine. Only the first is what the demo uses.

**The demo (ModalTests) never runs TileFoundry locally.** Inside the Modal container the image
installs an editable copy of an uploaded source snapshot:

```
add_local_dir("STAGING" -> "/opt/tilefoundry", copy=True)           # run_demo.py:180
cd /opt/tilefoundry && python -m pip install --no-deps --no-build-isolation -e . -e targets/nvidia  # run_demo.py:182
```

That yields `0.0.0+demo.47a6ed6a7dc8` from `/opt/tilefoundry`, pinned to the same commit
`1393b19`. No `schedule` subcommand is ever invoked by any demo entrypoint, and no bundle in
`results/` contains TIR — the demo stops at HIR analysis (`Fused.txt`/`Naive.txt` are
`tilefoundry analyze` output, not IR).

**Locally there is an editable install of the same commit** — this is what performed the
conversion, and it needed no changes:

| | |
|---|---|
| interpreter | `/usr/bin/python3.12` (3.12.3) |
| venv | `/home/shijingchang/workspace/audit/tilefoundry-1026/venv` |
| `tilefoundry` | `/home/shijingchang/workspace/TileFoundry/src/tilefoundry/__init__.py` |
| `tilefoundry_nvidia` | `/home/shijingchang/workspace/TileFoundry/targets/nvidia/tilefoundry_nvidia/__init__.py` |
| CLI | `/home/shijingchang/workspace/audit/tilefoundry-1026/venv/bin/tilefoundry` |
| PEP 610 | `{"dir_info":{"editable":true},"url":"file:///home/shijingchang/workspace/TileFoundry"}` |

The conda envs `mk`, `triton`, `wheel-build` have **no** `tilefoundry`. `PYTHONPATH`-only from
source does **not** work: `isl`, `apache-tvm-ffi`, `graphviz`, `ortools` are missing from every
system interpreter (only `jinja2`, `torch`, `networkx`, `sympy` exist in `wheel-build`).
The existing editable install was reused rather than building a new environment.

## 2. The actual command

```bash
cd /home/shijingchang/workspace/ModalTests/tilefoundry_demo/cases/rmsnorm_fp8

/home/shijingchang/workspace/audit/tilefoundry-1026/venv/bin/python -m tilefoundry.cli \
  schedule finalize "rms_norm_quant_opt_scheduled.py:FusedScheduled" \
  rms_norm_quant_opt.tir
```

`SOURCE` is `FILE.py[:Module[.function]]` (`cli/__init__.py:67-72`); `out` is a required
positional (`cli/__init__.py:232-237`). `--dim NAME=EXTENT` binds open dimensions — not needed
here, since this module has no `DimVar`. Exit code 0, and re-running produces a byte-identical
file.

**The selector was never the problem.** `Fused` and `Fused.rms_norm_quant` are both accepted
and both reach the same lowering: with no `.function` suffix `select_ir` returns the Module
(`cli/source.py:78-81`) and `@module(entry=...)` already names `rms_norm_quant`; with the suffix
it returns `replace(selected, entry=path[0])` (`cli/source.py:102-104`). `FusedScheduled` lowers
under either spelling too. Both spellings fail on the authored file for the reason in §3.

## 3. Why the authored file cannot be lowered

```
$ ... schedule finalize "rms_norm_quant_opt.py:Fused" /tmp/x.tir
tilefoundry: error: held has instruction candidates [T.copy, T.copy_async, T.copy_async_tensor,
T.copy_reduce_async_tensor, T.ldmatrix, T.stmatrix] but no automatic selection;
write tf.schedule to choose the instruction and its attributes
```

`schedule candidates` classifies every site in `Fused`:

| site | op | class |
|---|---|---|
| `held:59`, `scaling:60`, `<value>:68`, `<value>:69` | `tf.reshard` | **CHOICE** — `T.copy` is the only admitted candidate; lowering refuses to pick |
| `mean:62`, `<value>:65` | `tf.reduce` | **REFUSED-ALL** — `T.reduce` rejects the operand |
| everything else | cast/unary/binary/clamp | AUTO |

### Blocker A — ambiguous instruction selection

Six TIR carriers are registered for a reshard. `sole_candidate`
(`visitor_registry/candidates/__init__.py:83-98`) returns `None` when there are several, and
`IssuePlan.op` (`convert_hir_to_tir.py:1054-1070`) raises. The fix is to name the instruction:
`tf.schedule((src, dst), op=T.copy())`.

### Blocker B — `T.reduce` cannot take a cta-sharded operand

`T.reduce`'s operands are declared against `_ONE_MESH = MeshPattern(("thread",), …)`
(`ir/tir/reduce.py:40-50`). `MeshPattern` requires the picked topology set to **equal** the
pattern set (`ir/pattern/match.py:414-433`), so a `ShardLayout` over `Mesh((cta 132, thread 256))`
never matches:

```
ScheduleOp: T.reduce ... src: ShardLayout(... mesh=Mesh((Topology("cta", 132),
Topology("thread", 256)), ...)) does not match ShardLayout(Layout((*_,), (*_,)), _,
Mesh(('thread',), ComposedLayout(None, mesh_offset, ...)))
```

There is no way around it from the HIR side: `Reduce` is the only registered reduce carrier, and
the finalize pipeline is `target_passes(BEFORE_LOWER) + ConvertHIRToTir + target_passes(AFTER_LOWER)`
(`schedule/finalize.py:17-18`) — the sole CUDA pass is a `BEFORE_CODEGEN` host-entry synthesizer
that finalize never runs. There is no mesh-narrowing pass.

The one tolerated shape is `cta = Broadcast()` on both operands, because
`participant_layout` drops the non-participant level when every attr on it is a `Broadcast`
(`types/utils.py:19-74`, wired in at `pattern/match.py:92-110`). The authored file uses
`S(0)`/`S(1)` splits on the cta axis, so it does not qualify.

**This is the same pattern the shipped `tir/gated_norm_1024.py` uses**: the cta index appears
only inside the gmem `T.tensor_view` slice, and both reduce operands are thread-only
ShardLayouts.

### Blocker C — reshaping across a mesh region is not affine-lowerable

Fixing A and B surfaces the third: `T.reduce` needs the `(rows, blocks, block)` split so axis
−1 is the quantization block, but reshaping the row tensor crosses the mesh boundary:

```
non-affine address { [i0, i1] -> [i0, o1, o2, o3, i1 - 1792o1 - 128o2 - 4o3] : ... }
```

Probed in isolation: a `(2, 7168) -> (2, 56, 128)` reshape fails under a thread-only mesh
(`convert_hir_to_tir.py:1036`), and moving the reshape outside the mesh is refused as well
(`runs outside every mesh scope … move this operation inside with Mesh(...)`). A 3-D
`Reshape` node also cannot read a 2-D source through a scheduled `T.copy`
(`reads src.shape = dst.shape`).

## 4. What was done instead

`rms_norm_quant_opt_scheduled.py` keeps every constant, dtype, operator and ordering from the
authored file. Three things change, all forced by the above:

1. Each reshard is an explicit `tf.schedule(..., op=T.copy())` into a `tf.empty(... "rmem")`
   destination, with the destination layout written out as the 5-mode split
   `(PAIR @ m.x, 4 @ m.y, 14, 32 @ m.t, 4)`.
2. The CTA's rows are windowed off the global tensor with `a[m0:m0 + PAIR, :, :]` **before** the
   copy, so the cta index lives only in the pointer slice and both reduce operands end up
   thread-only.
3. The two parameters are **declared** `(ROWS, BLOCKS, BLOCK)` and `(1, BLOCKS, BLOCK)` instead
   of `(ROWS, H)` / `(1, H)`. The buffer is identical row-major memory: `(264, 7168)` and
   `(264, 56, 128)` are the same 1 892 352 elements. The authored file builds exactly this view
   at `tf.reshape(tf.cast(held, "f32"), (ROWS, BLOCKS, BLOCK))`.

`FP8_MAX`'s reciprocal is hoisted to a module constant (`INV_FP8_MAX`); the arithmetic is the
same, and the emitted TIR carries the same literal `0.002232142857142857`.

### Numerical check

TileFoundry's own evaluator refuses `MeshCoord` (`evaluator: MeshCoord is not modelled`), so
the original HIR was evaluated as the oracle and cross-checked against a torch transcription in
the emitted order:

```
original HIR quant : (264, 7168) torch.float8_e4m3fn
quant bit-identical : True      quant max|diff| : 0.0
scale bit-identical : True      scale max|diff| : 0.0
```

### TIR is genuinely verified, not merely printed

```python
m = load_authored_ir("rms_norm_quant_opt.tir:FusedScheduled")
verify_module(m)                       # OK
as_script(m) == open("...").read()     # True  — canonical text, byte-identical
```

This is the same contract the upstream TIR corpus enforces
(`test_schedule_corpus.py:403-407`).

## 5. HIR op → TIR node mapping

62 `T.*` call sites in the emitted TIR, of 10 distinct kinds: 18 `T.alloc_tensor`,
11 `T.tensor_view`, 11 `T.ptr_of`, 5 `T.binary`, 4 `T.fill`, 4 `T.copy`, 4 `T.cast`,
2 `T.unary`, 2 `T.reduce`, 1 `T.clamp`.

| HIR (`rms_norm_quant_opt.py`) | TIR node(s) | note |
|---|---|---|
| `@module(entry=…, target=_H200)` | `@module(...)` + `@prim_func(target=CudaTarget("nvidia.h200_sxm"), output_count=2)` | `Module`/`Function` → `PrimFunction`; the 2 outputs become `out`/`out_1` params |
| `Mesh(("cta",), (CTAS,))` | `Mesh((Topology("cta", 132),), layout=(132,), names=("tile",))` | preserved |
| `Mesh(("thread",), (PAIR,4,32))` | `Mesh((Topology("thread", 256),), layout=(2,4,32), names=("x","y","t"))` | preserved |
| `tf.zeros(...)` (both outputs) | `T.fill(out, 0.0)`, `T.fill(out_1, 0.0)` | |
| `tf.empty(Tensor[…,"rmem"])` | `T.alloc_tensor(tensor_type=Tensor[…, "rmem"])` | ×18; the register allocation |
| `tf.schedule((a[w], dst), op=T.copy())` | `T.tensor_view(T.ptr_of(a[cta.tile*2 : cta.tile*2+1, 0:0+1]), …)` + `T.copy(value, dst_view)` | the cta window lands in the pointer slice |
| `tf.cast(held, "f32")` | `T.cast(src_view, rows, dtype="f32")` | |
| `tf.square(rows)` | `T.unary(rows, value, kind=UnaryKind.SQUARE)` | |
| `tf.reduce(…, MEAN)` | `T.reduce(value_1, dst_view, axes=(1, 2), keepdim=True, kind=ReduceKind.MEAN)` | axes are 3-D here |
| `+ EPS` | `T.fill(scalar, 1e-06)` + `T.binary(…, kind=BinaryKind.ADD)` | |
| `tf.rsqrt(…)` | `T.unary(…, kind=UnaryKind.RSQRT)` | |
| `rows * rsqrt` | `T.binary(rows, value_3, value_4, kind=BinaryKind.MUL)` | |
| `tf.cast(…, "bf16")` | `T.cast(value_4, value_5, dtype="bf16")` | |
| `tf.mul(bf16, scaling)` | `T.binary(value_5, rhs_view, normed, kind=BinaryKind.MUL)` | bf16 product |
| `tf.cast(normed, "f32")` | `T.cast(normed, n32, dtype="f32")` | the block-scale is computed in f32 |
| `tf.reduce(…, ABS_MAX)` | `T.reduce(n32, dst_view_3, axes=(2,), keepdim=True, kind=ReduceKind.ABS_MAX)` | |
| `* (1.0 / FP8_MAX)` | `T.fill(scalar_1, 0.002232142857142857)` + `T.binary(…, kind=BinaryKind.MUL)` | |
| `blocks / scale` | `T.binary(n32, scale, value_6, kind=BinaryKind.DIV)` | |
| `tf.clamp(…, -448.0, 448.0)` | `T.clamp(value_6, value_7, min_val=-448.0, max_val=448.0)` | |
| `tf.cast(…, "fp8e4m3")` | `T.cast(value_7, quant, dtype="fp8e4m3")` | |
| `tf.reshard(…, "gmem")` | `T.tensor_view(T.ptr_of(out[…]), …)` + `T.copy(quant, window)` | |
| `tf.reshape` (HIR-level view) | **no node** | folded into the 3-D declared shape; no runtime cost |

**Two HIR constructs had no direct TIR counterpart and were resolved by restructuring**, not by
faking TIR: the cta-split reduce operand (now a windowed thread-only operand) and the
mesh-crossing reshape (now a declared rank).

## 6. Is the H200 static description sufficient?

**Yes — completely.** Nothing in this conversion needed a device:

```
$ ... target show nvidia.h200_sxm
device: nvidia.h200_sxm   digest: 82704400182e1a0b86f1c205f7e86a782bc2bd4418f03eaaea59ab499486cf8b
architecture: nvidia.sm90 digest: 97097b07d49435e3229e853618fd60cf6eee466db1469b78da7fd611252aa4d1
facts: capabilities: ['mma.sync','wgmma.mma_async','ldmatrix','stmatrix','cp.async', …]
```

`schedule finalize` only needs the target to resolve which instructions are admissible; it
performs instruction selection and layout projection, never codegen, launch or probe. No CUDA
driver call, no `nvidia-smi`, no Modal task was run. `finalize` deliberately runs only
`BEFORE_LOWER`/`AFTER_LOWER` passes, so even the `BEFORE_CODEGEN` CUTLASS/CMake path is untouched.

## 7. Minimal fix suggestion (upstream, if you want the authored file to lower)

Neither blocker is a bug — both are design choices — so "fixing" them means either a tool change
or an HIR rewrite. Cheapest first:

1. **HIR rewrite (no tool change, what was done here).** Name the four `T.copy` sites, window the
   CTA's rows before every reduce, declare the `(rows, blocks, block)` rank. ~40 lines in this
   case.
2. **Upstream convenience: let reshard default to `T.copy`** when it is the sole *admitted*
   candidate. `sole_candidate` already returns `None` purely on candidate count
   (`candidates/__init__.py:88-89`); counting admitted candidates instead would remove blocker A
   with no semantic change, since the other five carriers are refused on shape/storage/mesh
   anyway. This would not help B or C.
3. **Upstream: accept a Broadcast cta level.** Already works via `participant_layout`; the
   authored file just needs `B(...)` instead of `S(0)` on the cta axis for reduce operands. This
   is the only variant that keeps the `(cta, thread)` scope for the reduction.
4. **Not worth fixing: mesh-crossing reshape.** Making it affine-lowerable is a reasonably deep
   change to `convert_hir_to_tir`, and declaring the rank costs nothing at runtime.

None of this blocks the goal: the TIR exists and is verified.

---

### Files

- `rms_norm_quant_opt.py` — untouched.
- `rms_norm_quant_opt_scheduled.py` — new scheduled HIR (the input actually lowered).
- `rms_norm_quant_opt.tir` — generated TIR, 185 lines.
- The RMSNorm ModalTests E2E flow was not modified, and no GPU/Modal command was run.
