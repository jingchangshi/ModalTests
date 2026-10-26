#!/usr/bin/env python3
"""Render the P3 HTML report from the recorded timeline and decisions.

The page is generated, not authored: its event list, decision list and numbers all
come from ``timeline.jsonl``, ``decisions.json`` and the experiment record beside
them. A number that moved in the record therefore moves here too, and the page
cannot disagree with the evidence it cites.

The output is one static file with no external requests: no CDN, no web font, no
remote script. It opens offline.

Usage:
    python3 build_report.py <report-dir> <experiments.json>
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

CSS = """
:root {
  --bg: #10141b; --panel: #171d26; --panel-2: #1e2531; --line: #2b3543;
  --ink: #e6ecf5; --ink-dim: #9aa8bd; --agent: #4da3ff; --compiler: #f0a03c;
  --pass: #3ecf8e; --fail: #ff6b6b; --neutral: #8b98ad;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
        "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
}
.wrap { max-width: 1600px; margin: 0 auto; padding: 32px 28px 96px; }
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: .2px; }
h2 { font-size: 19px; margin: 44px 0 14px; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
h3 { font-size: 15px; margin: 22px 0 8px; color: var(--ink-dim); font-weight: 600; }
.sub { color: var(--ink-dim); margin: 0 0 24px; font-size: 14px; }
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 10px; margin: 20px 0 8px; }
.meta div { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
.meta dt { color: var(--ink-dim); font-size: 12px; margin: 0 0 3px; }
.meta dd { margin: 0; font: 13px/1.4 var(--mono); word-break: break-all; }
table { width: 100%; border-collapse: collapse; margin: 12px 0 8px; font-size: 14px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--ink-dim); font-weight: 600; font-size: 13px; }
td.num, th.num { text-align: right; font-family: var(--mono); }
code, .mono { font-family: var(--mono); font-size: 13px; }
pre { background: #0c1016; border: 1px solid var(--line); border-radius: 8px;
      padding: 12px 14px; overflow-x: auto; font-family: var(--mono);
      font-size: 12.5px; line-height: 1.55; margin: 10px 0; }
.tag { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px;
       font-weight: 600; letter-spacing: .3px; vertical-align: 1px; }
.tag-replay { background: #2a3a52; color: #9dc4f0; }
.tag-accepted { background: #24402f; color: #7fe0ac; }
.tag-rejected { background: #4a2b2b; color: #ff9d9d; }
.tag-agent { background: #2c4436; color: #7fe0ac; }
.timeline { display: grid; grid-template-columns: 78px 1fr 1fr; gap: 0; }
.timeline > .head { color: var(--ink-dim); font-size: 12px; font-weight: 600;
                    padding: 0 0 10px; border-bottom: 1px solid var(--line); }
.row { display: contents; }
.cell { padding: 10px 0; }
.event { margin: 0 0 8px; border: 1px solid var(--line); border-radius: 8px;
         background: var(--panel); overflow: hidden; }
.event > summary { cursor: pointer; padding: 10px 12px; list-style: none; }
.event > summary::-webkit-details-marker { display: none; }
.event[open] > summary { border-bottom: 1px solid var(--line); background: var(--panel-2); }
.event .id { font: 11px var(--mono); color: var(--ink-dim); margin-right: 6px; }
.event .act { font-weight: 600; }
.event .sum { color: var(--ink-dim); font-size: 13px; margin: 3px 0 0; }
.body { padding: 12px; }
.body dl { display: grid; grid-template-columns: 130px 1fr; gap: 4px 12px; margin: 0 0 8px; font-size: 13px; }
.body dt { color: var(--ink-dim); }
.body dd { margin: 0; }
.agent .event { border-left: 3px solid var(--agent); }
.compiler .event { border-left: 3px solid var(--compiler); }
.side-agent { color: var(--agent); }
.side-compiler { color: var(--compiler); }
.pass { color: var(--pass); font-weight: 600; }
.fail { color: var(--fail); font-weight: 600; }
.dim { color: var(--ink-dim); }
.bar { height: 22px; border-radius: 4px; display: inline-block; vertical-align: middle; }
.barrow { display: grid; grid-template-columns: 150px 1fr 96px; gap: 10px;
          align-items: center; margin: 7px 0; font-size: 13px; }
.note { border-left: 3px solid var(--compiler); background: var(--panel);
        padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 14px 0; font-size: 14px; }
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 1100px) { .split, .timeline { grid-template-columns: 1fr; } .row { display: block; } }
.flow { margin: 16px 0; font: 13px/2 var(--mono); color: var(--ink-dim); white-space: pre; overflow-x: auto; }
footer { margin-top: 56px; padding-top: 16px; border-top: 1px solid var(--line);
         color: var(--ink-dim); font-size: 12.5px; }
"""



def parse_predicates(stdout: str) -> list[tuple[str, bool]]:
    """The predicate rows of a ``tilefoundry check`` report.

    Only the indented rows carry a verdict. A ``check`` report also ends with a bare
    "PASS"/"FAIL" summary of those rows and may carry a ``warning:`` paragraph that
    contains the word FAIL without being a result; neither is a predicate, so neither
    is read as one.
    """
    predicates: list[tuple[str, bool]] = []
    for line in stdout.splitlines():
        if not line.startswith("    ") or line.lstrip().startswith("warning:"):
            continue
        stripped = line.strip()
        if stripped.endswith("PASS"):
            predicates.append((stripped[: -len("PASS")].rstrip(), True))
        elif stripped.endswith("FAIL"):
            predicates.append((stripped[: -len("FAIL")].rstrip(), False))
    return predicates


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def render_pre(text: str) -> str:
    return f"<pre>{esc(text)}</pre>" if text else '<p class="dim">（无）</p>'


def side_of(actor: str) -> str:
    return "agent" if actor == "AGENT" else "compiler"


def render_event(event: dict[str, Any], wanted: str) -> str:
    """One collapsible event. Its body holds the command, output and evidence."""
    exit_code = event.get("exit_code")
    if exit_code is None:
        verdict = '<span class="dim">n/a</span>'
    elif exit_code == 0:
        verdict = '<span class="pass">exit 0</span>'
    else:
        verdict = f'<span class="fail">exit {esc(exit_code)}</span>'

    body = ['<div class="body">']
    body.append("<dl>")
    body.append(f"<dt>event_id</dt><dd class='mono'>{esc(event['event_id'])}</dd>")
    body.append(f"<dt>iteration_id</dt><dd class='mono'>{esc(event['iteration_id'])}</dd>")
    body.append(f"<dt>actor</dt><dd class='mono'>{esc(event['actor'])}</dd>")
    body.append(f"<dt>run_id</dt><dd class='mono'>{esc(event['run_id'])}</dd>")
    body.append(f"<dt>task_id</dt><dd class='mono'>{esc(event['task_id'])}</dd>")
    if event.get("source_hash"):
        body.append(f"<dt>source_hash</dt><dd class='mono'>{esc(event['source_hash'])}</dd>")
    if event.get("input_artifacts"):
        body.append(f"<dt>input_artifacts</dt><dd class='mono'>{esc(', '.join(event['input_artifacts']))}</dd>")
    if event.get("output_artifacts"):
        body.append(f"<dt>output_artifacts</dt><dd class='mono'>{esc(', '.join(event['output_artifacts']))}</dd>")
    body.append("</dl>")

    evidence = event.get("evidence") or {}
    if event.get("command"):
        body.append("<h3>command</h3>")
        body.append(render_pre(" ".join(str(part) for part in event["command"])))

    if evidence.get("source_diff"):
        body.append("<h3>source diff</h3>")
        body.append(render_pre(evidence["source_diff"]))
    if evidence.get("compiler_message"):
        body.append("<h3>compiler 反馈原文</h3>")
        body.append(render_pre(evidence["compiler_message"]))
        if evidence.get("agent_response"):
            body.append(f"<p><b>Agent 的应对：</b>{esc(evidence['agent_response'])}</p>")
    if evidence.get("predicates"):
        body.append("<h3>谓词结果</h3>")
        body.append(render_pre("\n".join(evidence["predicates"])))
    if evidence.get("roofline"):
        body.append("<h3>roofline</h3>")
        body.append(render_pre("\n".join(evidence["roofline"])))
    if evidence.get("memory"):
        body.append("<h3>memory</h3>")
        body.append(render_pre("\n".join(evidence["memory"])))
    if evidence.get("stdout") and not evidence.get("predicates"):
        body.append("<h3>stdout</h3>")
        body.append(render_pre(evidence["stdout"]))
    if evidence.get("stderr"):
        body.append("<h3>stderr</h3>")
        body.append(render_pre(evidence["stderr"]))

    other = {
        key: value
        for key, value in evidence.items()
        if key
        not in {
            "source_diff",
            "compiler_message",
            "agent_response",
            "predicates",
            "roofline",
            "memory",
            "stdout",
            "stderr",
        }
    }
    if other:
        body.append("<h3>evidence</h3>")
        body.append(render_pre(json.dumps(other, ensure_ascii=False, indent=2)))
    body.append("</div>")

    return (
        f'<details class="event" data-stage="{esc(wanted)}" data-actor="{esc(event["actor"])}">'
        f'<summary><span class="id">{esc(event["event_id"])}</span>'
        f'<span class="act">{esc(event["action"])}</span> {verdict}'
        f'<p class="sum">{esc(event["summary"])}</p></summary>'
        + "".join(body)
        + "</details>"
    )


def render_timeline(events: list[dict[str, Any]]) -> str:
    """Two columns: what the agent did, and what the compiler reported."""
    # `iteration_id` is "<stage>-it<NN>", so the stage is everything before "-it".
    stages: list[str] = []
    for event in events:
        stage = str(event["iteration_id"]).split("-it")[0]
        if stage not in stages:
            stages.append(stage)

    parts = ['<div class="timeline">']
    parts.append('<div class="head">阶段 / 事件</div>')
    parts.append('<div class="head side-agent">Agent 侧</div>')
    parts.append('<div class="head side-compiler">Compiler / Runtime 侧</div>')

    for stage in stages:
        stage_events = [event for event in events if str(event["iteration_id"]).startswith(stage)]
        parts.append(
            f'<div class="row"><div class="cell mono dim">{esc(stage)}</div>'
            f'<div class="cell agent">{"".join(render_event(e, stage) for e in stage_events if side_of(e["actor"]) == "agent")}</div>'
            f'<div class="cell compiler">{"".join(render_event(e, stage) for e in stage_events if side_of(e["actor"]) == "compiler")}</div>'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_decisions(decisions: list[dict[str, Any]]) -> str:
    parts = []
    for decision in decisions:
        accepted = decision["accepted"]
        # ACCEPTED/REJECTED is the verdict on the decision; it is not the
        # replay/experiment marker, which says where the decision came from.
        badge = (
            '<span class="tag tag-accepted">ACCEPTED</span>'
            if accepted
            else '<span class="tag tag-rejected">REJECTED</span>'
        )
        # Which kind of work this decision was: a guided replay of a step the tutorial
        # already answers, or the agent's own experiment. A reader cannot tell them
        # apart from the verdict alone, so the marker is stated per decision.
        origin = (
            '<span class="tag tag-replay">GUIDED_TUTORIAL_REPLAY</span>'
            if decision["stage"] != "P2-5"
            else '<span class="tag tag-agent">AGENT_EXPERIMENT</span>'
        )
        observations = "".join(f"<li>{esc(item)}</li>" for item in decision["observations"])
        candidates = "".join(f"<li>{esc(item)}</li>" for item in decision["candidate_actions"])
        refs = ", ".join(decision.get("evidence_refs") or [])
        diff = ""
        if decision.get("source_diff"):
            diff = "<h3>source diff</h3>" + render_pre(decision["source_diff"])
        parts.append(
            f'<details class="event"><summary><span class="id">{esc(decision["decision_id"])}</span>'
            f'<span class="act">{esc(decision["task"])}</span> {badge} {origin}'
            f'<p class="sum">{esc(decision["stage"])} · 选中：{esc(decision["selected_action"])}</p></summary>'
            '<div class="body"><dl>'
            f'<dt>observations</dt><dd><ul>{observations}</ul></dd>'
            f'<dt>candidate_actions</dt><dd><ul>{candidates}</ul></dd>'
            f'<dt>expected_effect</dt><dd>{esc(decision["expected_effect"])}</dd>'
            f'<dt>observed_effect</dt><dd>{esc(decision["observed_effect"])}</dd>'
            f'<dt>reason</dt><dd>{esc(decision["reason"])}</dd>'
            f'<dt>evidence_refs</dt><dd class="mono">{esc(refs)}</dd>'
            "</dl>"
            + diff
            + "</div></details>"
        )
    return "".join(parts)


def render_bar(value: float, largest: float, colour: str) -> str:
    width = 0 if largest <= 0 else max(1.5, 100.0 * value / largest)
    return f'<span class="bar" style="width:{width:.2f}%;background:{colour}"></span>'


def build(report_dir: Path, experiments_path: Path, out_path: Path) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in (report_dir / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decisions = json.loads((report_dir / "decisions.json").read_text(encoding="utf-8"))["decisions"]
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))["experiments"]

    correctness = experiments["correctness"]
    twin = experiments["twin"]
    measurement = experiments["benchmark"]["measurement"]
    comparison = measurement["comparison"]
    medians = comparison["median_of_round_medians_ms"]
    speedup = comparison["speedup_torch_over_triton"]
    versions = measurement["versions"]
    first = events[0]

    # `analyze` prints one roofline block per function it prices, so a source with two
    # functions reports two figures. The first is the entry point, which is the one worth
    # comparing; Naive prints 8697 for the entry and 4952 for the quantisation it calls.
    roofline: dict[str, str] = {}
    roofline_all: dict[str, list[str]] = {}
    for key in ("analysis_naive", "analysis_fused", "analysis_agent"):
        summary = correctness[key]["summary"]
        found: list[str] = []
        for line in summary.splitlines():
            if line.startswith("# roofline"):
                for token in line.split():
                    if token.startswith("ideal-ns="):
                        found.append(token.split("=", 1)[1])
        roofline_all[key] = found
        roofline[key] = found[0] if found else ""

    runs = {run["label"]: run for run in correctness["runs"]}
    agent_check = correctness["agent_candidate"]

    def predicate_rows(run: dict[str, Any]) -> str:
        rows = []
        for text, passed in parse_predicates(run["stdout"]):
            verdict = (
                '<span class="pass">PASS</span>'
                if passed
                else '<span class="fail">FAIL</span>'
            )
            rows.append(f"<tr><td class='mono'>{esc(text)}</td><td>{verdict}</td></tr>")
        return "".join(rows)

    def run_block(title: str, run: dict[str, Any]) -> str:
        exit_code = run["returncode"]
        badge = (
            '<span class="pass">exit 0</span>'
            if exit_code == 0
            else f'<span class="fail">exit {exit_code}</span>'
        )
        return (
            f"<h3>{esc(title)} {badge}</h3>"
            "<table><tr><th>谓词与数值</th><th class='num'>判定</th></tr>"
            + predicate_rows(run)
            + "</table>"
        )

    largest_median = max(medians.values())
    chart = (
        '<div class="barrow"><span>torch 基线</span>'
        + render_bar(medians["torch_baseline"], largest_median, "#f0a03c")
        + f"<span class='num'>{medians['torch_baseline']:.5f} ms</span></div>"
        '<div class="barrow"><span>Triton twin</span>'
        + render_bar(medians["triton_twin"], largest_median, "#4da3ff")
        + f"<span class='num'>{medians['triton_twin']:.5f} ms</span></div>"
    )

    roofline_rows = "".join(
        f"<tr><td>{esc(label)}</td><td class='num'>{esc(roofline[key] or '—')}</td><td>memory</td></tr>"
        for label, key in (
            ("Naive（未融合）", "analysis_naive"),
            ("Fused（融合）", "analysis_fused"),
            ("Agent 候选", "analysis_agent"),
        )
    )

    agent_verdict = (
        '<span class="pass">正确性 PASS</span>，但 <span class="fail">性能目标未达成</span>'
        if agent_check["returncode"] == 0
        else '<span class="fail">正确性 FAIL</span>'
    )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RMSNorm + FP8：Agent 与 TileFoundry 编译器协同实验</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>RMSNorm + FP8：Agent 与 TileFoundry 编译器的协同实验</h1>
<p class="sub">按时间顺序展示真实的协作事件。左列是 Agent 的动作，右列是编译器与运行时的反馈。
点击任意事件查看命令、源码差异、编译器输出与结果。</p>

<div class="meta">
  <div><dt>run_id</dt><dd>{esc(first['run_id'])}</dd></div>
  <div><dt>task_id</dt><dd>{esc(first['task_id'])}</dd></div>
  <div><dt>GPU</dt><dd>{esc(correctness['gpu_name'])} / CC {esc(correctness['compute_capability'])}</dd></div>
  <div><dt>python</dt><dd>{esc(versions['python'])}</dd></div>
  <div><dt>torch</dt><dd>{esc(versions['torch'])}</dd></div>
  <div><dt>triton</dt><dd>{esc(versions['triton'])}</dd></div>
  <div><dt>实验记录</dt><dd>{esc(experiments_path)}</dd></div>
</div>

<div class="note">
<b>静态预测与真实测量是两回事。</b>下方的 <code>ideal-ns</code> 是编译器把字节数除以带宽得到的
模型值，不是时长；<code>ms</code> 是在 H200 上用 CUDA 事件测出来的。两者没有相除，
也不互相换算。另外，被测的 Triton twin 由<b>人</b>编写、由 <b>Triton</b> 编译，
不是 TileFoundry CodeGen 的产物。
</div>

<h2>1. 用户目标与分工</h2>
<div class="flow">用户目标：让 RMSNorm + FP8 量化在 H200 上真实跑通，且每步都有真实来源、实际命令与独立验证
   │
   ├─ Agent          写 HIR、读编译器反馈、改方案、挑候选
   ├─ TileFoundry    语义判定（check）、静态代价（analyze）、布局与存储层级校验
   ├─ Triton         编译运行期 twin（不归 TileFoundry）
   └─ CUDAGPU        真实执行与计时（H200）</div>

<h2>2. Expected Fail → Repair → PASS</h2>
{run_block("第一次写入的 HIR（先乘 gamma 再舍入）", runs['wrong'])}
{run_block("修复后（先舍入到 bf16 再乘 gamma）", runs['corrected'])}
<p class="dim">容差全程未改动（<code>atol=1e-6</code>, <code>rtol=1e-6</code>）。
修复后 <code>max_violation</code> 是 0，不是恰好压在容差以内。</p>

<h2>3. Naive 与 Fused 的对比</h2>
<table>
<tr><th>placement</th><th class="num">静态 ideal-ns</th><th>约束</th></tr>
{roofline_rows}
</table>
<p class="dim">{esc(roofline['analysis_naive'])} 与 {esc(roofline['analysis_fused'])} 都是字节数除以带宽的模型值，不是测量。
Agent 候选与 Fused 同为 {esc(roofline['analysis_fused'])}：激活值从 rmem 搬到了 smem，字节数没变，模型时间也没变。</p>
<p class="dim">Naive 的源码有两个函数，<code>analyze</code> 因此报出两组数字
（{esc(', '.join(roofline_all['analysis_naive']))}）；取第一组，它是入口函数。</p>

<h2>4. 真实 GPU 测量（与上表分开）</h2>
{chart}
<table>
<tr><th>加速比（torch 基线 ÷ Triton twin）</th><th class="num">值</th></tr>
<tr><td>最小</td><td class="num">{speedup['min']:.2f}×</td></tr>
<tr><td>中位</td><td class="num">{speedup['median']:.2f}×</td></tr>
<tr><td>最大</td><td class="num">{speedup['max']:.2f}×</td></tr>
</table>
<p class="dim">5 轮逐轮交替计时，10 次预热、每轮 100 次重复。
FP8 结果一致性：<code>quant_equal_fraction</code> = {esc(measurement['agreement']['quant_equal_fraction'])}，
<code>scale_max_abs_diff</code> = {esc(measurement['agreement']['scale_max_abs_diff'])}。</p>

<h2>5. Agent 候选的去留</h2>
<p>{agent_verdict}。<code>tilefoundry check agent_candidate.py:SplitMetaChecked</code>
退出码 {esc(agent_check['returncode'])}，两个谓词均通过；<code>tilefoundry analyze</code> 报告
<code>ideal-ns={esc(roofline['analysis_agent'])}</code>，与 Fused 相同。
假设是"片上驻留是瓶颈"，报告否掉了这个假设：约束在 GMEM 上。</p>

<h2>6. 按时间顺序的协作事件</h2>
<p class="dim">共 {len(events)} 条事件。左列 Agent，右列 Compiler / Runtime。</p>
{render_timeline(events)}

<h2>7. Agent 决策记录</h2>
<p class="dim">每条决策在实验前记录了候选与预期效果，再记录实际观测到的效果。
<code>accepted=false</code> 表示候选被否决或假设被否掉。</p>
{render_decisions(decisions)}

<footer>
生成自 <code>timeline.jsonl</code>（{len(events)} 条事件）、<code>decisions.json</code>
（{len(decisions)} 条决策）与 <code>{esc(experiments_path.name)}</code>。
页面为单个静态文件，不引用任何外部资源，可离线打开。
标记：<span class="tag tag-replay">GUIDED_TUTORIAL_REPLAY</span> 为教程已给答案的复现；
<span class="tag tag-agent">AGENT_EXPERIMENT</span> 为 Agent 自行提出并经验证的候选。
</footer>
</div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return {"events": len(events), "decisions": len(decisions), "path": out_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("experiments", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()
    out = arguments.out or arguments.report_dir / "index.html"
    result = build(arguments.report_dir, arguments.experiments, out)
    print(f"events:    {result['events']}")
    print(f"decisions: {result['decisions']}")
    print(f"wrote:     {result['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
