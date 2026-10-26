# RMSNorm + FP8：Agent 与 TileFoundry 编译器的协同实验报告

本报告记录一次完整的 Agent–Compiler 协同实验。同目录的 `timeline.jsonl`、`decisions.json`
和 `index.html` 是本次实验的结构化记录，报告中的每一条结论都能在这三个文件里找到对应事件。

标记约定：

- `GUIDED_TUTORIAL_REPLAY`：官方教程已经给出答案，本次按教程复现并独立验证。
- `AGENT_EXPERIMENT`：Agent 在本次实验中自行提出、由编译器独立判定的候选。

## 0. 实验身份

| 项目 | 值 |
| --- | --- |
| 仓库 | `jingchangshi/ModalTests`，分支 `feat/tilefoundry-rmsnorm-fp8` |
| 实验用例 | `tilefoundry_demo/cases/rmsnorm_fp8/` |
| 实验记录 | `results/experiments-20261026-110746-743687/experiments.json` |
| GPU | NVIDIA H200，compute capability (9, 0)，目标架构 SM90 |
| Python | 3.12.15 |
| torch | 2.9.1+cu128，CUDA 12.8 |
| Triton | 3.5.1 |
| 随机种子 | 0 |

用例源码的 SHA-256 附在 `timeline.jsonl` 每条事件的 `source_hash` 字段中。GPU 型号、
驱动版本和库版本取自真实运行容器的返回记录，不是本地推断。

## 1. 用户交给 Agent 的原始任务

> 让 RMSNorm 加 FP8 分块量化的这一步在 H200 上真实跑通，并且每一步都要有真实来源、
> 实际命令、真实反馈和独立验证。

这条任务有两个约束，它们决定了后面所有决策：

1. **必须能跑**：结论要有 GPU 上的实测支撑。
2. **不能只是"跑通"**：要让读者看到 Agent 读到什么、编译器说了什么、Agent 因此改了什么。

## 2. Agent 读取了哪些模型与硬件事实

Agent 在写第一版 HIR 之前读取的事实全部来自 `config.json` 与用例目录，没有从教程正文
抄写常量：

| 事实 | 值 | 来源 |
| --- | --- | --- |
| 行数 | 2（检查用输入）或 264（分析用形状） | `x.pt` 张量形状 |
| 隐藏维 `H` | 7168 | `config.json` |
| 量化块宽 `BLOCK` | 128 | `config.json` 的 `quantization_config.weight_block_size[1]` |
| 块数 `BLOCKS` | 56 | `7168 / 128` |
| RMSNorm 的 `eps` | 1e-6 | `config.json` 的 `rms_norm_eps` |
| FP8 最大值 | 448.0 | `fp8e4m3` 的最大有限值 |
| 输出 dtype | `fp8e4m3` 激活 + `f32` 缩放 | `expected.pt` |
| 目标 GPU | NVIDIA H200 / SM90 | 任务给定的远端环境 |

这里有一个真实的坑：检查用的 `x.pt` 只有 **2 行**，而教程分析用的形状是 **264 行**。
两者不是同一个形状，所以在 P2-5 里同一个模块不能同时用于检查和性能分析（见第 6 节）。

## 3. TileFoundry 提供了哪些可调用的编译器服务

本次实际调用过三个服务，每个都有独立的退出码语义：

| 服务 | 作用 | 退出码 |
| --- | --- | --- |
| `tilefoundry check` | 按谓词逐一比较实际输出与 `expected.pt` | 0 全通过 / 1 有失败 / 2 用法错误 |
| `tilefoundry analyze` | 输出静态代价报告：`--compute-cost`、`--memory`、`--roofline` | 0 正常 |
| HIR 类型推导与合法性检查 | 在 `check` / `analyze` 内部运行，负责任务级布局与存储层级校验 | 失败即报错退出 |

`check` 可用的谓词包括 `allclose`、`cosine`、`equal`、`max_abs`、`max_rel`、`nan_inf`、
`rel_l2`、`ulp`。注意 `nan_inf` 在 fp8 上必然失败，本次没有使用它。

`analyze` 只接受源码和报告相关的开关，**不接受** `--inputs`、`--weights`、`--device`。
这一点在实验中造成过一次真实报错，已记录在 `decisions.json`。

## 4. Agent 第一次写出的 HIR

`GUIDED_TUTORIAL_REPLAY`

Agent 按"我记忆中的 RMSNorm + 量化"写出了第一版，即 `rms_norm_quant.py` 中的 `Wrong`
类。它的算术顺序是：

```python
rows = tf.cast(a, "f32")
mean = tf.reduce(tf.square(rows), (-1,), True, ReduceKind.MEAN)
normed = tf.cast(rows * tf.rsqrt(mean + EPS) * tf.cast(gamma, "f32"), "bf16")  # 先乘 gamma
blocks = tf.reshape(tf.cast(normed, "f32"), (ROWS, BLOCKS, BLOCK))
scale = tf.reduce(blocks, (-1,), True, ReduceKind.ABS_MAX) * (1.0 / FP8_MAX)
quant = tf.cast(tf.clamp(blocks / scale, -FP8_MAX, FP8_MAX), "fp8e4m3")
```

关键错误在第 3 行：**先乘 `gamma`，再舍入到 bf16**。

## 5. 编译器怎样发现问题

`GUIDED_TUTORIAL_REPLAY`

`tilefoundry check rms_norm_quant.py:Wrong` 的真实输出（H200 容器内执行，退出码 1）：

```
  output[0]   fp8e4m3[2,7168]   ref_norm 19084.5
    equal                              mismatched 432 elements 14336 FAIL
  output[1]   f32[2,56]   ref_norm 0.0685877
    allclose(atol=1e-06 rtol=1e-06)    max_violation 6.87445e-05  FAIL
```

这是**语义反馈**。它的信息量不在"失败"两个字，而在失败的样子：

- `equal` 谓词在 14336 个元素中有 **432 个**精确不相等。如果只是数值噪声，fp8 量化的
  结果不会有精确相等这种判据，更不会只有 3% 的元素错。
- `allclose` 的 `max_violation` 是 6.87e-05，比 `atol=1e-06` 大约 69 倍。

两个谓词同时失败，且 `equal` 的失配集中在少数元素上，说明这是一个**结构性错误**，
而不是精度不够。

## 6. Agent 怎样利用错误反馈修复

`GUIDED_TUTORIAL_REPLAY`

`decisions.json#D001` 在实验前就记录了候选动作和预期效果：

- 候选一：放宽 `atol`，让它通过。
- 候选二：把 bf16 舍入移到 `gamma` 乘法之前。
- 选中：候选二。
- 预期效果：**在不改动容差的前提下**两个谓词都通过。

选候选二的理由是：`gamma` 是每通道的缩放因子，先乘再做 bf16 舍入，等于对"已经缩放的
值"做量化，得到的每一行与模型定义的那一步不是同一个数。这是一个可以预判的差值，不是
随机噪声，所以放宽容差只会掩盖错误。

修复后的代码（`RmsNormQuant`）：

```python
normed = tf.cast(rows * tf.rsqrt(mean + EPS), "bf16") * gamma   # 先舍入，再乘 gamma
```

`tilefoundry check rms_norm_quant.py:RmsNormQuant` 的真实输出（退出码 0）：

```
  output[0]   fp8e4m3[2,7168]   ref_norm 19084.5
    equal                              mismatched 0 elements 14336 PASS
  output[1]   f32[2,56]   ref_norm 0.0685877
    allclose(atol=1e-06 rtol=1e-06)    max_violation 0            PASS
```

容差从头到尾没有改动。修复后 `max_violation` 是 **0**，不是"刚好压在容差以内"。

## 7. Agent 怎样利用内存分析决定融合

`GUIDED_TUTORIAL_REPLAY`

### 7.1 计算代价反馈：不是优化目标

Agent 先问"是不是算得太多"。`analyze` 的 roofline 报告对两个版本的结论都是
`bound-by=memory`。两个不同的实现都被内存约束，说明算术量不是这两个版本的差异所在，
把它当优化目标不会有收益。

### 7.2 内存反馈：这才是差异所在

| 指标 | Naive | Fused |
| --- | --- | --- |
| roofline `ideal-ns` | 8697 | 3755 |
| 约束 | memory | memory |
| GMEM 读（`@logical`） | 7.23 MiB | 3.62 MiB |
| GMEM 写（`@logical`） | 5.47 MiB | 1.86 MiB |

`analyze` 对每个函数各打一组数字，而 Naive 的源码里有两个函数，所以它报出两组
（8697 和 4952）；上表取的是入口函数那一组。

两者相差 **3.61 MiB 读 + 3.61 MiB 写**。精确地说是 3,784,704 字节，等于
`264 × 7168 × 2`，也就是被归一化后那一份中间结果：未融合时它要写回 GMEM 再读回来，
融合后它留在片上。

`decisions.json#D002` 记录的预期效果是"GMEM 读写各减去中间结果的往返"，观测结果是
"3,784,704 字节，`ideal-ns` 从 8697 降到 3755"，两者一致，所以接受融合。

需要注意的是：`ideal-ns` 是**静态预测**，是把字节数除以带宽得到的模型值，不是时长。
它不能和任何毫秒数直接比较。

## 8. 为什么需要 Runtime Twin

`GUIDED_TUTORIAL_REPLAY`

到第 7 节为止，全部证据都是静态的。`ideal-ns` 回答"这个形状在这台机器上的下界大概是
多少"，不回答"这段代码实际跑多久"。要拿到实测，必须有能在 GPU 上真正执行的代码。

HIR 的 `analyze` 不产生可执行产物，所以实测需要一个运行时的对照实现，也就是 twin。

## 9. Twin 的代码由谁编写，编译由谁执行

这是本报告最需要明确区分归属的一节。

| 环节 | 归属 |
| --- | --- |
| Twin 的 Triton 源码 | 人工编写，`cases/rmsnorm_fp8/twin.py` |
| Twin 的编译 | **Triton 3.5.1**，不是 TileFoundry CodeGen |
| Twin 的正确性判定 | TileFoundry `check` 的谓词，对 `expected.pt` |

`timeline.jsonl#E008` 的 `actor` 是 `TRITON_COMPILER`，并在 `evidence.attribution` 里写明
"TRITON_COMPILER, not TileFoundry codegen"。把 Triton 的编译结果算到 TileFoundry 的
CodeGen 名下是错的，本报告不做这件事。

Twin 在 H200 上的真实检查输出（退出码 0）：

```
  output[0]   fp8e4m3[2,7168]   ref_norm 19084.5
    equal                              mismatched 0 elements 14336 PASS
  output[1]   f32[2,56]   ref_norm 0.0685877
    allclose(atol=1e-06 rtol=1e-06)    max_violation 0            PASS
```

这里也暴露过一个真实的形状 bug：`twin.py` 一开始用模块常量 `ROWS = 264` 推导行数，而
检查输入是 2 行。改为从 `a.shape[0]` 取行数后才通过。这和第 2 节提到的形状分裂是同一
个根源。

## 10. 正确性验证由谁提供

`GUIDED_TUTORIAL_REPLAY`

| 被验证对象 | 验证者 | 基准 |
| --- | --- | --- |
| 错误 HIR（`Wrong`） | `tilefoundry check` | `expected.pt` |
| 修复后 HIR（`RmsNormQuant`） | `tilefoundry check` | `expected.pt` |
| Triton twin | `tilefoundry check` | `expected.pt` |
| Agent 候选（`SplitMetaChecked`） | `tilefoundry check` | `expected.pt` |
| torch 基线 | 无，它就是基准 | — |

四条路径共用同一份 `expected.pt` 和同一组谓词，所以它们的"通过"是同一个意义下的通过。
PyTorch 基线不参与验证，它是被测量的对照项。

## 11. 性能测量提供什么新信息

`GUIDED_TUTORIAL_REPLAY`

`benchmark.py` 在 H200 上用 CUDA 事件计时，10 次预热、每轮 100 次重复、5 轮，两臂
**逐轮交替**执行。真实结果（`results/experiments-20261026-110746-743687/`）：

| 项 | torch 基线 | Triton twin |
| --- | --- | --- |
| 各轮中位数的中位数 | 0.107824 ms | 0.025664 ms |

| 加速比 | 值 |
| --- | --- |
| 最小 | 4.09× |
| 中位 | 4.18× |
| 最大 | 4.23× |

`timeline.jsonl#E009` 记录了两个臂的完整分布、`measured_region`、`seed` 和库版本。

为什么必须逐轮交替、为什么必须报区间：**同一个会话内**基线每轮中位数在 0.1007 ms 到
0.1473 ms 之间移动，而会话之间的移动更大——另一次运行测到基线 0.11872 ms、双胞胎
0.026272 ms，加速比 4.44× 到 4.58×。如果两臂是先后测完的，设备温度的漂移会整体算到
其中一臂头上；如果只报一个比值，就宣称了测量本身不支持的精度。所以本次报区间，并把
会话之间的差异单独说明。

测量还给出了一项静态模型给不出的信息：**两条路径的 FP8 结果一致**。`quant_equal_fraction`
= 0.99999946，`scale_max_abs_diff` = 0.0。也就是说 twin 与基线之间 4 倍多的差距不是靠
"少算东西"换来的，量化的数值基本重合。这个结论 static 模型无法提供。

必须写清楚这不是什么：本次没有测"TileFoundry 生成的代码 vs PyTorch"。测的是人手写的
Triton twin vs PyTorch。`ideal-ns` 也没有被除以这些毫秒数——那是两个不同的模型，相除
得到的数字没有意义。`experiments.json` 中对应字段是
`not_a_speedup_claim_from_static_model`。

## 12. Agent 最终怎样选择候选

`AGENT_EXPERIMENT`

这是本次唯一由 Agent 自行提出、并由编译器独立判定的候选。

### 12.1 推理链（记录于 `p2-5-agent-exploration.json`）

1. 两个教程版本都 `bound-by=memory`，算术不是杠杆（来自编译器报告）。
2. 两者唯一的结构差异是 `@func` 边界：未融合版本把归一化结果写回 GMEM 再读回，
   差值 3,784,704 字节（来自编译器报告）。
3. 融合版本已经消掉了这份往返，所以"再消一次同样的流量"没有空间。
4. 融合版本仍然付出的是**片上驻留**：激活值整个 func 期间留在 rmem，
   `rmem r70.57MiB/w56.07MiB@logical`。假设是：把激活值挪到 smem 这一层能不能换来时间。

### 12.2 编译器五次拒绝

候选不是一次写对的。每一次拒绝都是真实报错原文：

| 次数 | 编译器反馈 |
| --- | --- |
| 1 | `Binary: input 0 is laid out more coarsely and kept in rmem, which does not reach the units this runs on; reshard it to smem or gmem first` |
| 2 | `Binary: operands have conflicting storage (rmem, smem); a multi-input op requires its concrete operands to share one residency` |
| 3 | `Reshape cannot express the sharded layout: new shape does not align with the input layout factorization` |
| 4 | `default_shard_layout: logical axis 0 size 2 is not divisible by mesh extent product 4 (role 'layout')` |
| 5 | `evaluator: shape '[264, 56, 128]' is invalid for input of size 14336` |

每一次的读法和改法都记在 `p2-5-agent-exploration.json` 的
`compiler_feedback_encountered` 里，并与 `timeline.jsonl#E013`–`E017` 对应。要点：

- 第 1 次说明自定义的 4-D 布局比乘法所在单元更粗，Agent 于是退回融合版本的 3-D 分片。
- 第 2 次说明多输入算子必须共用一个驻留层级，Agent 于是把 `gamma` 也移到 smem。
- 第 3、4 次是形状分裂的直接后果：2 行的检查形状不能让 CTA 轴承载真正的切分，
  而 264 行的线程网格按 `PAIR × 4 = 8` 切分行，除不尽 2。Agent 因此把"分析用 264 行"
  和"检查用 2 行"拆成两个模块变体。
- 第 5 次是 Agent 自己没把行数对齐，属于低级错误，编译器直接指出。

### 12.3 结果：正确，但没有更快

`tilefoundry check agent_candidate.py:SplitMetaChecked` 真实输出（H200，退出码 0）：

```
  output[0]   fp8e4m3[2,7168]   ref_norm 19084.5
    equal                              mismatched 0 elements 14336 PASS
  output[1]   f32[2,56]   ref_norm 0.0685877
    allclose(atol=1e-06 rtol=1e-06)    max_violation 0            PASS
```

`tilefoundry analyze agent_candidate.py:SplitMeta` 的报告：

| 项 | 融合版本 | Agent 候选 |
| --- | --- | --- |
| `ideal-ns` | 3755 | **3755** |
| 约束 | memory | memory |
| GMEM | `r3.62MiB/w1.86MiB` | `r3.62MiB/w1.86MiB`（相同） |
| rmem | `r70.57MiB/w56.07MiB` | `r16B/w0` |
| smem | 无 | `r70.57MiB/w56.07MiB` |

激活值确实从 rmem 搬到了 smem，字节数一点没少（70.57 / 56.07 MiB 原样平移），而
`ideal-ns` **一点没降**。假设被同一份报告否掉了：约束在 GMEM，不在片上。片上换层
不改变 GMEM 流量，所以不改变模型时间。

`decisions.json#D005` 的 `accepted` 是 **false**。这个候选正确、被编译器接受，但它没有
达成它自己的性能目标。把它写成"Agent 优化了 kernel"是造假，本报告不这么写。

## 13. 五种 Compiler Feedback 的对应位置

| 类型 | 真实来源 | 位置 |
| --- | --- | --- |
| 语义反馈 | `equal` 432/14336 失配，`allclose` max_violation 6.87e-05 | `timeline.jsonl#E002`，本报告第 5 节 |
| 计算代价反馈 | `bound-by=memory`（两个版本都是） | `timeline.jsonl#E005`、`E006`，本报告 7.1 节 |
| 内存反馈 | GMEM 读 7.23→3.62 MiB，写 5.47→1.86 MiB，差 3,784,704 B | `timeline.jsonl#E005`、`E006`、`E007`，本报告 7.2 节 |
| 后端正确性反馈 | Triton 3.5.1 编译 + `check` 对 `expected.pt` 比较 + H200 真实执行 | `timeline.jsonl#E008`，本报告第 9 节 |
| 性能反馈 | 5 轮交替计时，4.09×–4.23×，中位 4.18×；会话间漂移单独说明 | `timeline.jsonl#E009`、`E010`，本报告第 11 节 |

另有一类不在上述五类中的反馈，本次真实出现且信息量最大：**布局与存储层级反馈**，
即第 12.2 节的五次拒绝。它既不是语义错误也不是性能问题，而是"这个写法在目标硬件上
不成立"。把它归到上面任何一类都会失真，因此单列。

## 14. 哪些环节仍需人工设计或预先给定实现

必须如实列出，否则会把指导式复现说成自主发现：

1. **任务本身、常量、基准答案由人给定**：`H`、`BLOCK`、`eps`、`FP8_MAX` 来自
   `config.json`，`expected.pt` 由 `make_reference.py` 预先算好。
2. **融合这一决策来自教程**：第 7 节是 `GUIDED_TUTORIAL_REPLAY`。Agent 用编译器报告
   独立验证了融合的理由，但"要融合"这个方向不是它自己找的。
3. **Triton twin 由人编写**：第 9 节已说明，编译归 Triton，源码归于人工。
4. **两个模块变体的分裂由人决定**：264 行分析与 2 行检查必须分开，这个结构是人工
   拆的，不是 Agent 自动推导出来的。
5. **验收阈值由人给定**：`atol=1e-6`、`rtol=1e-6`、`PAIR=2`、`CTAS=132` 都是预设值。

## 15. 哪些环节尚不属于真正的自动算子优化闭环

1. **没有自动搜索空间**。本次只试了少数几个有依据的候选，不是一个自动的 placement
   搜索（没有枚举、没有 beam、没有代价驱动的迭代）。
2. **没有自动改写**。Agent 的修改是逐次手写 HIR 片段，编译器只判定"接受/拒绝"，
   不生成改写。
3. **无法自动决定从哪个形状检查**。第 12.2 节第 3–5 次拒绝全部源于 2 行与 264 行的
   分裂，这说明检查形状与优化形状的对齐仍需人工介入。
4. **性能反馈还没有回流到 placement 搜索**。第 11 节测的是人手写的 twin，这个数字
   没有用于给第 12 节的候选排序；目前 placement 的取舍只看静态 `ideal-ns`。
5. **本轮 Agent 候选的净结果是零收益**。闭环是否有效，本次没有取得一个正向样例。

## 16. 结论

1. 教程的语义陷阱是真实的：先乘 `gamma` 再舍入会产生 432/14336 个精确失配，
   重排顺序后 `max_violation` 归零。
2. 融合的收益是可量化、可复现的：GMEM 读写各减 3,784,704 字节，`ideal-ns` 8697→3755，
   两个版本都 `bound-by=memory`。
3. Triton twin 在 H200 上比 torch 基线快 4.09×–4.23×（中位 4.18×），且量化结果基本
   一致（`quant_equal_fraction` = 0.99999946）。这是**实测**，与 `ideal-ns` 是两个模型。
4. Agent 自提出的第三个 placement 被编译器接受、正确性通过，但 `ideal-ns` 没有下降，
   假设被否。这条负结果按原样记录。
5. 本次最强的一类编译器反馈不在五类之中：布局与存储层级的拒绝，共 5 次真实报错。
   它说明编译器不只是"判对错"，还能否掉在目标硬件上不成立的写法。

## 附：本报告所依据的文件

| 文件 | 内容 |
| --- | --- |
| `timeline.jsonl` | 17 条结构化事件，含命令、退出码、证据 |
| `decisions.json` | 10 条决策，含实验前记录的候选与预期效果 |
| `index.html` | 可离线浏览的 HTML 报告，Agent 与 Compiler 分列 |
| `results/experiments-20261026-110746-743687/experiments.json` | 本报告引用的原始实验记录 |
| `cases/rmsnorm_fp8/p2-5-agent-exploration.json` | P2-5 的推理链与五次拒绝全文 |
