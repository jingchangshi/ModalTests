# AGENTS.md — Triton-on-Modal 编译研究工作区

本目录（仓库根 `modal/`）是一个 Triton GPU 编译器研究工作区：在本地编辑 Triton kernel 与测试，
通过 Modal 平台在远程 GPU 上编译、运行并保存各阶段编译产物（IR dump）。

本文件供自动化 agent 在此仓库内工作前阅读。改动本文件时保持中英术语一致、命令可直接复制执行。

---

## 1. 目录结构

```text
modal/                          # 仓库根
├── AGENTS.md                   # 本文件
└── triton/                     # 所有工作都在这个子目录内进行（modal run 必须在此目录执行）
    ├── modal_env.py            # Modal App/Volume/Image 定义（被所有入口共享）
    ├── study_compiler.py       # 主入口：编译研究运行（结构化元数据 + 产物打包）
    ├── run_test.py             # 辅助入口：简单回归运行（较少使用）
    ├── study_pytest_plugin.py  # pytest 插件：采集环境/pipeline/计时信息
    ├── kernels/                # Triton kernel 源码（add.py, softmax.py, fused_attention.py）
    ├── tests/                  # pytest 测试用例（test_add.py, test_softmax.py, test_fused_attention.py）
    ├── README.md               # 面向人的说明（英文）
    └── triton-study-<run-id>/  # 从 Volume 下载回来的产物包（已 gitignore）
```

约定：

- kernel 放 `triton/kernels/<name>.py`，对应测试放 `triton/tests/test_<name>.py`。
- `modal_env.py` 的 image 通过 `add_local_python_source("modal_env", "study_pytest_plugin",
  "kernels", "tests")` 把这四个模块打包上传，所以新增 kernel/测试**不需要**改 image 定义；
  但若新增顶层目录/模块（如 `kernels/utils.py` 属于 `kernels` 包则没事，新建顶层包则需要），
  必须同步修改 `modal_env.py`。
- `triton-study-*` 目录与 `*.tar.gz` 已在 `.gitignore` 中，不要提交。

## 2. 环境（本地）

- 本地没有 GPU、没有 `nvidia-smi`；**不要**在本地安装/运行 torch、triton，也**不要**在本地
  执行 pytest——编译与运行只发生在远程 Modal 容器内。
- 本地只需 `modal` CLI。使用 conda 环境 `triton`：

  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate triton
  ```

  该环境里只装了 `modal`（client 1.5.5）；`python` 是 3.11。
- Modal 认证 profile 为 `jingchangshi`（`modal profile current` 可查）。若命令报未登录，
  先运行 `modal profile activate jingchangshi`。
- 网络注意：本机到 pypi.org 很慢（几百 KB/s 以下），但 PyPI 清华镜像
  （`https://pypi.tuna.tsinghua.edu.cn/simple/...`）很快。需要查 triton wheel 内部文件时，
  用镜像下载（约 248 MB，十几秒可完成），再 `python -c "import zipfile; ..."` 解包检查。

## 3. 远程运行环境（Modal 容器，由镜像决定，勿在本地复现）

远程容器规格（从 study 产物 `results/environment.json` 读取，2026-09 实测）：

| 项 | 值 |
|---|---|
| GPU | NVIDIA A10（24 GB），compute capability **8.6**（sm86） |
| Python | 3.11.12（容器内） |
| torch | 2.14.0+cu130 |
| triton | **3.8.0** |
| CUDA | 13.0 |
| pytest | 9.1.1 |

镜像定义在 `triton/modal_env.py`：`debian_slim(python_version="3.11")` +
`uv_pip_install torch triton numpy pytest`。改镜像后第一次运行会重新构建，耗时更长。

### Triton 3.8.0 libdevice 注意事项（重要，写 kernel 前先读）

- 正确的导入方式（官方测试同款）：

  ```python
  import triton.language.extra.libdevice as tld   # CUDA 后端编译时自动映射到 CUDA 实现
  ```

  原理：wheel 里 `triton/language/extra/libdevice.py` 只有 `...` 桩（无实现）；
  真正实现是 `triton/language/extra/cuda/libdevice.py`（`@core.extern` + `extern_elementwise`）。
  `CUDABackend.get_module_map()` 在编译期把 `triton.language.extra.libdevice` 这个模块名映射到
  CUDA 版实现，所以上述 import 在 CUDA 后端下有效且可移植；不要直接
  `import triton.language.extra.cuda.libdevice`（GPU 后端可移植性差）。
- 函数名与 libdevice 符号映射（fp32 路径）：`tld.exp→__nv_expf`、`tld.fma→__nv_fmaf`、
  `tld.div_rn→__nv_fdiv_rn`、`tld.pow→__nv_powf`、`tld.tanh→__nv_tanhf`、
  `tld.fast_expf→__nv_fast_expf` 等。完整清单看 wheel 内
  `triton/language/extra/cuda/libdevice.py`（用第 2 节的镜像方法解包查看）。
- libdevice 函数是严格类型分派的：fp32 输入必须整体是 fp32。Python 标量会按
  `semantic.to_tensor_type` 规则转类型（float→fp32；若 |x| 超出 fp32 范围会变 fp64 导致
  dispatch 失败）。`other=float("-inf")` 这类写法是安全的。
- 判断 libdevice 是否真的进了产物：TTIR 里会出现
  `tt.extern_elementwise ... symbol = "__nv_*"`，PTX 里会出现 `ex2.approx.ftz.f32`
  （exp）、`fma.rn.ftz.f32`（fma）、`div.rn.ftz.f32`（div_rn）等指令
  （`ftz` 来自默认 `enable_reflect_ftz=True`）。

## 4. 运行命令（在 `triton/` 目录内执行）

```bash
conda activate triton
cd <仓库根>/triton

# 标准编译研究运行（推荐）：stage 级产物 + 结构化元数据
modal run study_compiler.py --test-path tests/test_softmax.py --detail stage

# pass 级：额外 dump 每个 MLIR pass 边界的 IR 与 LLVM IR（日志很大，慎用）
modal run study_compiler.py --test-path tests/test_softmax.py --detail pass

# 旧入口（简单回归，产物较少，一般不用）
modal run run_test.py --test-path tests/test_add.py
```

- `--test-path` 接 pytest 相对路径（相对 `triton/`），一次一个文件；`--detail` 只接受
  `stage` 或 `pass`。
- 运行结束屏幕会打印 JSON 结果和下载命令，形如：

  ```bash
  modal volume get triton-artifacts studies/<run-id>/triton-study-<run-id>.tar.gz ./triton-study-<run-id>.tar.gz
  ```

  直接复制执行即可把产物包下载到当前目录；解包后得到 `triton-study-<run-id>/` 目录。
- 典型耗时：单 kernel stage 级运行约 1–2 分钟（含 JIT 约 1.7 s + pytest + 打包上传）。
  长时间无输出属正常，`study_compiler.py` 的容器 timeout 是 1800 s。

## 5. 产物包结构与阅读顺序

```text
triton-study-<run-id>/
├── manifest.json            # 先看这个：状态、环境、pipeline、kernel_metrics、产物 sha256 清单
├── dumps/<CACHE_KEY>/       # 每个 kernel 编译产物的 cache key 目录
│   ├── <kernel>.ttir        # ① Triton IR：看 tt.extern_elementwise 的 symbol 是否为预期 __nv_*
│   ├── <kernel>.ttgir       # ② TritonGPU IR：看 layout（blocked encoding）、warp 数
│   ├── <kernel>.llir        # ③ LLVM IR：看 libdevice 是否被 inlining（__nv_*.exit 基本块）
│   ├── <kernel>.ptx         # ④ PTX：看最终指令（ex2.approx / fma.rn.ftz / div.rn.ftz ...）
│   ├── <kernel>.cubin       # ⑤ SASS 容器
│   └── <kernel>.sass        # ⑥ 反汇编
├── results/
│   ├── environment.json     # GPU/驱动/版本
│   ├── pipeline.json        # 编译 pipeline 阶段与 CUDAOptions（含 libdevice.10.bc 路径）
│   ├── kernel_metrics.jsonl # 每行一个 kernel：cold/warm 计时、max_error、自定义字段
│   └── pytest_wall_times.jsonl
└── logs/
    └── compiler.log         # pytest 全输出 + MLIR/LLVM pass timing 报告
```

计时语义（解读 metrics 时注意，别把 cold 当纯编译时间）：

- `cold_jit_and_first_run_ms`：首次启动墙钟，含 JIT 编译 + CUDA 模块加载 + 首次执行。
- `warm_runtime_ms`：`triton.testing.do_bench` 的热运行时间。
- 编译各 pass 耗时看 `logs/compiler.log` 里的 "Pass execution timing report"。

## 6. 新增测试用例的固定套路

以 `tests/test_softmax.py` / `kernels/softmax.py` 为模板（已验证可跑通）：

1. kernel 放 `kernels/`，只依赖 `triton`、`triton.language`、`triton.language.extra.libdevice`。
2. 测试放 `tests/`，结构照抄 `test_softmax.py`：
   - 顶部 `from kernels.<name> import <kernel>`；
   - `_record_study_metrics(**metrics)` 写 `kernel_metrics.jsonl`（study 模式才有
     `TRITON_STUDY_RESULTS_DIR` 环境变量，本地无此变量时跳过）；
   - cold 计时包裹第一次 kernel 启动 + `torch.cuda.synchronize()`；warm 用
     `triton.testing.do_bench`；
   - 数值校验用 `torch.testing.assert_close`（fp32 建议 rtol=1e-5/atol=1e-5），
     并额外打印/记录 `max_error`。
3. **不要**给 kernel 传非 constexpr 的运行时标量参数：当前实测 pipeline 里 constexpr
   会被内联成常量（TTIR 中直接出现常量），运行时参数只有指针。需要多配置就多写几个
   `tl.constexpr` 组合或多个测试函数。
4. grid 设计：一维 grid + 每 program 处理整行（行长 ≤ BLOCK_SIZE）是最简单可靠的模式；
   跨行归约（如 softmax）用 `tl.max/tl.sum(..., axis=0)` 行内归约，避免跨 program 同步。
5. masked load 的 `other` 值要选对归约语义：softmax 用 `other=float("-inf")`
   （max 归约的恒等元），不要用 0。
6. 本地只能做静态检查：`python -m py_compile`、AST 解析、以及（如需）对照 wheel 内
   `triton/language/__init__.py` 与 `extra/cuda/libdevice.py` 确认用到的 API 名存在。
   正确性只能靠远程运行验证——跑一次 stage 级 study 看 manifest 的 `status` 和
   `max_error`。

## 7. 已知坑

- **本目录名就叫 `triton/`**：`cd triton` 后 `import triton` 不会因此坏掉
  （Modal 上传的是 `kernels`/`tests` 包，且容器内路径不同），但本地任何直接
  `import triton` 的脚本都必须在 `conda activate triton` 且**不在** `triton/` 目录内运行，
  否则会把本目录当包导入。agent 生成临时 python 脚本时放 `/tmp` 再执行。
- `tests/` 里的文件既是 pytest 用例也被 Modal 作为包上传，顶部不要有昂贵的模块级
  GPU 操作（GPU 逻辑放测试函数里）。
- `study_pytest_plugin.py` 的 pipeline hook 契约与 Triton 3.8 的
  `add_stages_inspection_hook` 绑定（hook_key 版本号 v2）。升级 Triton 大版本时，
  先跑一次 stage 级 study 并检查 `results/pipeline.json` 是否还产出真实阶段记录，
  若只有 warning 记录说明 hook 签名变了，需要同步改插件。
- `manifest.json` 的 `pytest_returncode` 非 0 时 `status=failed`，但产物照常打包上传——
  排查失败先看 `logs/compiler.log` 末尾的 pytest 报错，再看 dumps 是否有部分产物。
- Modal Volume `triton-artifacts` 按幂等覆盖写 `studies/<run-id>/`；run-id 精确到秒，
  同秒重跑会互相覆盖，重跑前确认上一个 run 已结束。

## 8. 已验证的基线（用于回归对比）

`tests/test_softmax.py`（4096×256 fp32，BLOCK_SIZE=256，libdevice exp/fma/div_rn），
run `20260912-141700`，sm86：

- status: passed，max_error ≈ 2.98e-08
- cold JIT + first run ≈ 1764 ms；warm ≈ 0.0219 ms
- 编译链证据：TTIR 含 `tt.extern_elementwise(symbol="__nv_expf"/"__nv_fmaf"/"__nv_fdiv_rn")`；
  PTX 含 `ex2.approx.ftz.f32`、`fma.rn.ftz.f32`、`div.rn.ftz.f32`；`.reqntid 128`（4 warps）。

`tests/test_add.py`（98432 元素，BLOCK_SIZE=256），run `20260912-131330`：
passed，max_error 0.0，cold ≈ 1665 ms，warm ≈ 0.0071 ms。

`tests/test_fused_attention.py`（FA v2 前向，BATCH=1×N_HEADS=2×N_CTX=1024×HEAD_DIM=64 fp16
causal，教程 06 v3.8.0 移植），run `20260912-154624`，sm86：

- status: passed，max_error ≈ 4.88e-04（教程同款 atol=1e-2，fp16 精度正常）
- cold ≈ 29747 ms（autotuner 实际编译了 6 个 config：warps∈{4,8} × stages∈{2,3,4}，
  其中 4 个落盘完整 dumps，1 个空目录属正常——编译失败被 autotuner 剔除）；warm ≈ 0.0225 ms
- 实测性能 ≈ 11.9 TFLOPS（A10 fp16 tensor core 理论峰值 125 TFLOPS 的 ~10%，
  小 batch+短序列的正常水平，教程 benchmark 用 batch4×heads32 才能到 100+ TFLOPS）
- 编译链证据（sm86 路径）：TTGIR 含 `ttg.nvidia_mma{versionMajor = 2, instrShape = [16, 8]}`
  + `ttg.async_copy_global_to_local`（cp.async 软件流水）；PTX 含
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`（64 或 128 条，随 BLOCK_N 不同）、
  `cp.async.*`×36、`ex2.approx`×36、813 个 b32 寄存器、`.reqntid 256`；SASS 含
  `HMMA.16816.`×64 + `LDGSTS.E.BYPASS.128`×16
- sm86 适配要点（与教程的差异）：
  - `supports_host_descriptor()` 需要 cc≥9，Ampere 走 fallback：直接传张量指针，
    kernel 内 `_maybe_make_tensor_desc` 创建 device 端 descriptor，由 TTIR pass
    `rewrite_tensor_descriptor_to_pointer`（capability<9 门控）降为普通指针运算；
    device 端 descriptor 仍需 `triton.set_allocator` 提供 global scratch。
  - `tl.range(..., warp_specialize=...)` 只在 Blackwell pipeline 生效，sm86 上是
    no-op（仅设置一个 op attr），保留不影响。
  - 研究版去掉了教程的 backward/benchmark harness（perf_report）和 autograd
    （`_attention` 是普通类而非 `torch.autograd.Function`），只保留前向。
- FA 运行时间构成参考：pytest 总墙钟 31.2 s ≈ 6 次编译的 MLIR/LLVM pass 累计 12.8 s
  （最贵：ConvertTritonGPUToLLVM 1.13 s、TritonGPUCoalesceAsyncCopy 0.76 s、
  TritonGPURemoveLayoutConversions 0.55 s）+ ptxas×6 + autotuner bench 循环 + torch 导入。

新用例跑完后建议把基线追加到本节。
