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
│   ├── environment.json     # CPU/GPU/版本等 run-level 静态环境信息
│   ├── pipeline.json        # 编译 pipeline 阶段与 CUDAOptions（含 libdevice.10.bc 路径）
│   ├── kernel_metrics.jsonl # 每行一个 kernel：cold/warm 计时、max_error、自定义字段
│   └── test_runtime.jsonl   # 每个 pytest case 的 wall time + 测试前后 loadavg
└── logs/
    └── compiler.log         # pytest 全输出 + MLIR/LLVM pass timing 报告
```

计时语义（解读 metrics 时注意，别把 cold 当纯编译时间）：

- `cold_jit_and_first_run_ms`：首次启动墙钟，含 JIT 编译 + CUDA 模块加载 + 首次执行。
- `warm_runtime_ms`：`triton.testing.do_bench` 的热运行时间。
- 编译各 pass 耗时看 `logs/compiler.log` 里的 "Pass execution timing report"。
- CPU 型号、逻辑核数等静态宿主信息由 `study_pytest_plugin.py` 自动写入 `environment.json`；每个 testcase 前后的 1/5/15 分钟 loadavg 自动写入 `test_runtime.jsonl`。具体 kernel/test 不要重复采集这些基础设施信息。

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

---

## 9 · tilefoundry_demo（跨服务器可运行的 TileFoundry-on-Modal 演示）

> 本节起是**另一个子项目**，与第 1–8 节的 `triton/` 工作区相互独立：两条代码路径、两套
> 入口、两套依赖。第 2 节里的 conda 环境名与 Modal profile 名是 `triton/` 那台机器的本地
> 约定，**不适用于本节**；本节的要求只写在这里。

### 9.1 定位与目录

```text
tilefoundry_demo/                 # 在 tilefoundry_demo/ 目录内执行 modal run
├── run_demo.py                   # 主入口：冻结本地 checkout → 上传镜像 → HIR 分析 → H200 编译/执行/比对 → 归档证据包
├── run_experiments.py            # RMSNorm + FP8 教程实验驱动（复用 run_demo 的镜像）
├── demo_container.py             # 容器内 stage 编排与证据采集
├── demo_task.py                  # 任务规格校验、stage/case/source 白名单
├── demo_report.py                # 证据归约：每个 stage 一个状态
├── demo_source.py / demo_build_inputs.py / demo_provenance.py
├── demo_pytest_plugin.py         # 容器内 pytest 观察插件
├── verify_bundle.py              # 校验下载回来的证据包
└── tasks/*.json                  # 任务注册表（stage 列表 + 期望结果 + 限额）
```

### 9.2 机器上只需要什么（跨服务器成立）

| 需要 | 说明 |
|---|---|
| git | 克隆本仓库与并排的 TileFoundry checkout |
| python3（客户端 3.11+） | 只用来跑 `modal` 客户端；venv 足够 |
| `modal` 客户端 | `python -m pip install modal` |
| 网络 | 访问 Modal API |
| 有效的 Modal 账号 | 已登录；H200 容量可用 |

不需要（这几点是本节的核心，改动时不要收回）：

- **no local GPU**：不需要 GPU，也不需要 `nvidia-smi`；所有设备操作都在远端 H200 容器里。
- **no local CUDA**：不需要本地 CUDA / nvcc / CMake / CUTLASS。
- **no local PyTorch**：不需要本地 PyTorch / transformers。
- **no local TileFoundry**：不需要本地安装；checkout 只是被打包进镜像。
- 不需要旧服务器的 conda 目录，也不需要任何 conda 环境。

Modal profile 名是任意的：只要 `modal profile current` 成功即可。换一台机器用
`modal token new`（或 `modal setup`）建任意名字的 profile 都能跑；本仓库不读
`MODAL_PROFILE`，也不要求某个特定名字。

```bash
python3 -m venv "$HOME/.venvs/tilefoundry-modal"
source "$HOME/.venvs/tilefoundry-modal/bin/activate"
python -m pip install modal
modal profile current          # 成功即可，名字无所谓
```

### 9.3 TileFoundry checkout 的解析顺序

源码在**模块导入时**被冻结、摘要并上传，所以路径必须在 `modal run` 之前确定。解析顺序仅此三条
（`run_demo.resolve_tilefoundry_repo()`）：

1. `--tilefoundry-repo PATH`（命令行参数）；
2. `$TILEFOUNDRY_REPO`（环境变量，非空时）；
3. 并排自动发现（sibling）：**本仓库旁边**名为 `TileFoundry` 的目录，即
   `<workspace>/TileFoundry` 与 `<workspace>/ModalTests` 并列——正是 9.4 那种布局，
   所以那里 `export TILEFOUNDRY_REPO` 是双保险而非必需。候选路径由 `run_demo.py` 自身位置
   推出（相对 `tilefoundry_demo/` 上两级），因此从任何工作目录执行结果都一样；
   **任何候选都不读当前工作目录**。

候选目录必须含 `pyproject.toml` 才算数。三条都不成立时，在**任何镜像构建之前**直接失败，
不产生 GPU 费用、不上传任何东西，报错首行是：

```text
no TileFoundry checkout was found, so no Modal image was built.
```

推荐用环境变量而不是参数：源码在导入时冻结，与冻结快照不一致的参数会被拒绝，而环境变量
在导入时就已生效。该路径必须是真正的 git clone，因为 manifest 记录 commit、分支、dirty
状态以及 CUTLASS 子模块 SHA。

### 9.4 运行命令（在 `tilefoundry_demo/` 内执行）

```bash
export TILEFOUNDRY_REPO="$HOME/workspace/TileFoundry"
cd "$HOME/workspace/ModalTests/tilefoundry_demo"

modal run run_demo.py --cpu-only                      # 无 GPU 预检：改依赖后先跑这个
modal run run_demo.py                                 # 默认任务 tasks/square.json（1×H200）
modal run run_demo.py --task tasks/square_cpu.json    # CPU-only 任务
modal run run_demo.py --run-id my-run-1               # 指定 run id，否则自动生成
modal run run_experiments.py --experiment all         # correctness + twin + benchmark
```

- `--cpu-only` 只跑到 preflight 为止；它证明镜像可装、两个包可导入、H200 target 已注册，
  但**不**声称任务完成（有 GPU stage 时 `task_status` 为 `NOT_RUN`）。
- 容器内固定目录 `/opt/tilefoundry`、`/opt/cutlass/include`、`/artifacts`、`/tmp`、
  `/tmp/tilefoundry-demo`、`/cases` 只存在于容器内部，**不是**宿主机的路径依赖。
- 下载与校验：运行结束会打印 `modal volume get` 命令；下载后用
  `python verify_bundle.py <archive> --result results/<run-id>/remote-result.json` 校验，
  输出 `VERIFY OK` 才说明归档、manifest、判决三者一致。

### 9.5 stage 与状态词汇

任务规格从 `tasks/*.json` 选 stage，stage 类型是**注册表白名单**，任务不能自定义新类型：
`environment`、`hir_analyze`、`hir_check`、`gpu_test`、`gpu_benchmark`、`archive`。
`demo_task.assert_runner_support()` 在加载任务时执行：本 runner 没有执行器的 stage 类型，
或注册在**别的 runner** 名下的 GPU case，都会在构建任何镜像、分配任何 GPU 之前作为本地
`SystemExit` 被拒绝，而不是执行一半再报成"运行过"。

每个 stage 独立归约，只可能是下面之一：

| 状态 | 含义 |
|---|---|
| `PASS` | 该 stage 自己的证据支持它 |
| `FAIL` | 跑过了，证据与声明矛盾 |
| `ERROR` | 跑了但产不出可判定的观测，或直接抛异常 |
| `BLOCKED` | 缺少它无法满足的前提（例如没有 CUDA 设备） |
| `NOT_RUN` | 没跑；`overall_reason` 说明缺哪一个、为什么 |
| `EXPECTED_FAIL_CONFIRMED` | 按任务声明**预期失败**且证据吻合；这是"确认了预期失败"，不是 `PASS`，绝不声称 kernel 跑通 |

`overall_status` 取**已有判决的 stage** 里最差的状态（`PASS` 与 `EXPECTED_FAIL_CONFIRMED`
并列最好）。状态常量以 `demo_report.py` 为准，文档不另写一套；本地测试会核对文档与代码一致。

`archive` **也会被判定**，只是发生在最后：tarball 真的写进 Volume 并记下摘要与大小之后，它变成
`PASS`，理由形如 `archived 4024663 bytes as runs/<run-id>/...tar.gz`。这里有一个由顺序带来的
细节：归档**内部**那份 `manifest.json` 是在归档存在之前写的，所以它必然记的是归档前的
`overall_status`——正因如此，`verify_bundle.py` 对这一条 stage 用运行打印出来的结果信封重算，
而不是相信归档里的副本。

证据里的 stage 键名是**任务自己的 stage id**，不是固定的五个旧名字：`tasks/square.json` 报
`environment`/`analyze`/`square`/`archive`，`tasks/square_cpu.json` 报
`environment`/`analyze`/`check`/`archive`。同一份 `manifest.json` 还带 `task_stages`：按执行
顺序每个 stage 一条，含 `stage_id`、`type`、`command`、`executed`。

此外它把**完整校验过的任务规格**原样存在 `task` 键下，与 `task_id`、`task_spec_sha`、
`task_stages`、`completion_scope` 并列。这是归档"自描述"的依据：`verify_bundle.py` 只凭下载
回来的归档就能重算判决——它按 `task` 声明的 stage 集合裁剪证据，而不是按检查器里写死的固定
清单，所以一个归档无法事后被重新解释成别的任务。

旧的五个键
（`environment_status`、`analysis_status`、`compile_status`、`execution_status`、
`correctness_status`）为兼容保留，但已改为**由任务的 stage 派生**——所以 CPU-only 任务把设备
相关的键报成 `NOT_RUN`，而不是继承一个它从未请求过的判决。

`completion_scope`（`full` / `cpu_only`）与 stage 状态是两回事，前者只说本次调用的覆盖范围。
`--cpu-only` 会写 `completion_scope=cpu_only`，并把每个声明了的 GPU stage 记为 `NOT_RUN`，
原因字符串是 `this invocation does not allocate a GPU, so a device stage was not run`：
有 GPU stage 时 `task_status` 为 `NOT_RUN`，纯 CPU 任务为 `COMPLETE`。

### 9.6 RMSNorm + FP8 参考数据（不提交，每轮在 Modal CPU 容器内重生）

`x.pt`、`expected.pt`、`model.safetensors` **不在 git 里**（`cases/rmsnorm_fp8/.gitignore`
已忽略），每次运行都在 Modal CPU 容器内重新生成，且**先于任何 GPU 申请**：

1. 只上传 case 的**源码**到该 run 专属前缀 `experiments/<run-id>/cases/rmsnorm_fp8/`：
   `config.json`、`make_reference.py`、`rms_norm_quant.py`、`rms_norm_quant_opt.py`、
   `twin.py`、`benchmark.py`，以及存在时的 `agent_candidate.py`。不读、不拷贝上一台机器
   产生的任何数据——三个参考数据文件**故意不在**这个清单里。
2. 一个 Modal **CPU** Function（没有 `gpu=`）在私有 `/tmp` 工作目录里跑
   `make_reference.py`，并按 case 契约校验产物：`x.pt` `[2, 7168]` bf16、
   `expected.pt` 为 `[quant (2, 7168) float8_e4m3fn, scale (2, 56) float32]` 两元素、
   `model.safetensors` 的 `gamma` `[1, 7168]` bf16；同时记录每个文件的 SHA-256、
   `make_reference.py` 与 `config.json` 的摘要、脚本写出的 provenance、以及容器内
   torch/transformers/safetensors 的实际版本。
3. 生成并校验通过后才调用 GPU Function：它重新读该 run 的 manifest、把 case 拷进私有目录、
   并逐个复核参考数据的摘要，因此不可能拿别的 run 的数据做基准。

生成失败时记成一条 error 记录，主机随后以 `SystemExit` 停止，文本含
`reference generation failed` 与 `No GPU was allocated and no experiment was run.`；
GPU Function **绝不会被调用**（`tests/test_run_experiments.py` 断言此时唯一的远端调用就是
生成器本身）。所以新机器上不需要本地 torch，也不需要从旧机器拷贝任何参考数据。
`--run-id` 可显式命名 run，否则自动生成；本地记录落在
`results/experiments-<run-id>/experiments.json`。

一次已验收的真实运行（run `20261029-082303-087224`）作为参照，不是对未来运行的承诺：
`x.pt` 30015 B / `4d3206a53565d4ce…`，`expected.pt` 16685 B / `7adfcdf1207747c9…`，
`model.safetensors` 14416 B / `4d3e4270115742c0…`；生成器摘要
`make_reference.py` `207b6379beef8f9e…`、`config.json` `67ba5386a7d28983…`；容器内
torch `2.9.1+cu128`、transformers `5.14.1`、safetensors `0.8.0`。

> 注意：这个演示容器里的 Triton 是 **3.5.1**，与第 3 节 `triton/` 子项目那套 3.8.0
> 不是同一个环境，不要把两节的版本号互相套用。

benchmark 记录同时写下测得的数与对照的基线（100 次重复、10 次 warmup、
`torch.cuda.Event` 计时）：torch 基线中位 `0.130160 ms`（p10 `0.127296` / p90 `0.138496`），
Triton twin 中位 `0.030288 ms`（p10 `0.028864` / p90 `0.032928`）；并记录与参考数据的一致性
`quant_equal_fraction` `0.9999995`、`scale_max_abs_diff` `0.0`。这些数不构成性能预测，只说明
跑它的那块卡实测到了什么、跟什么比。

`cases/rmsnorm_fp8/tutorial-provenance.json` 与 `cases/rmsnorm_fp8/reference-provenance.json`
是**历史记录**（上一次运行留下的），不要改写、也不要当成当前轮的 provenance；新运行会在
自己生成的数据旁边写自己的记录。

### 9.7 本地测试（无需 GPU、无需 torch、无需真实 Modal 客户端）

```bash
cd "$(git rev-parse --show-toplevel)/tilefoundry_demo"
python -m pytest tests -q
```

这套测试是纯 CPU 的，和第 2 节对 `triton/` 的"不要本地跑 pytest"约定**不同**：`triton/`
的测试要在远端 GPU 上跑，而 `tilefoundry_demo/` 的本地测试用 stub 顶替 Modal，断言的是
"证据不能被伪造"这类不变量，所以在本地跑是正确做法。

**唯一的前置条件**：`run_demo.py` 在 import 时就冻结源码快照（9.3），而下列测试文件在
模块级 import 它，所以本机必须能解析到一个 TileFoundry checkout：
`tests/test_run_demo.py`、`tests/test_checkout_resolution.py`、
`tests/test_task_runner_match.py`。`export TILEFOUNDRY_REPO=...` 或按 9.3 并排放一个
`TileFoundry/` 即可；否则只有这三个文件会在 import 阶段报
`no TileFoundry checkout was found`，其余测试文件照常通过。这个 checkout 不会被安装，
只被读取和摘要。

其中 `tests/test_no_machine_paths.py` 是本节能跨服务器的可执行保证：它扫描被 git 跟踪的
运行时文件（根目录所有 `*.py`、`tasks/*.json`、RMSNorm case 源码与 `config.json`、
`README.md`），拒绝任何 `/home/<user>` 字符串、任何 conda 路径或激活、任何 `modal profile`
名字要求、任何文档命令里的绝对 `cd`；同时**显式允许**容器内部路径与两份历史 provenance
文件；并拒绝任何被跟踪的参考数据二进制。改本节或 README 时先让它变绿。

### 9.8 与 `triton/` 子项目的关系

| | `triton/` | `tilefoundry_demo/` |
|---|---|---|
| 入口 | `modal run study_compiler.py` | `modal run run_demo.py` |
| 远端产物 | IR dump 压缩包 → Volume | 证据 tar.gz → Volume + 本地 `results/` |
| 本地 pytest | 不要跑（必须远端 GPU） | 必须跑（纯 CPU、stub 掉 Modal） |
| 本机环境 | 需要那台机器的 conda 环境与 profile | 任意 Linux + venv + `modal` 客户端 |

两者不要互相套用对方的运行约定；唯一共享的是"Modal 平台上跑远端 GPU"这件事本身。
