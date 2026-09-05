#!/usr/bin/env python3
"""Optional Gradio UI for the Dify or offline presales assistant."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_presales_lab.dify_client import DifyClient, DifyClientError
from ai_presales_lab.evaluation import load_cases
from ai_presales_lab.knowledge import KnowledgeBase
from ai_presales_lab.offline_engine import OfflineSolutionEngine

ROOT = Path(__file__).resolve().parents[1]


def _format_markdown(response) -> str:
    lines = [
        f"### 方案摘要\n{response.executive_summary}",
        f"\n**模型模式：** `{response.model_name}`",
        f"\n**审核状态：** `{response.review_status}`",
    ]
    if response.recommendation:
        lines.append("\n### 建议\n" + "\n".join(f"- {item}" for item in response.recommendation))
    if response.risks:
        lines.append(
            "\n### 风险\n"
            + "\n".join(
                f"- **{item.severity}** {item.description}（{item.action}）"
                for item in response.risks
            )
        )
    if response.clarifying_questions:
        lines.append(
            "\n### 待确认问题\n" + "\n".join(f"- {item}" for item in response.clarifying_questions)
        )
    if response.evidence:
        lines.append(
            "\n### 证据\n"
            + "\n".join(
                f"- `{item.evidence_id}` {item.title}：{item.excerpt}" for item in response.evidence
            )
        )
    return "\n".join(lines)


def build_app(mode: str):
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(
            "Install the optional UI dependency with: python3 -m pip install -e '.[demo]'"
        ) from exc

    cases = {case.case_id: case for case in load_cases(ROOT / "data/evaluation/cases.jsonl")}
    engine = OfflineSolutionEngine(KnowledgeBase(ROOT / "data/knowledge"))
    dify = DifyClient()

    def analyze(case_id: str):
        brief = cases[case_id]
        try:
            response = engine.analyze(brief) if mode == "mock" else dify.chat(brief)
            return _format_markdown(response), response.to_dict()
        except DifyClientError as exc:
            return f"### 调用失败\n`{exc}`", {"error": str(exc)}

    with gr.Blocks(title="AI Presales Lab") as app:
        gr.Markdown("# AI 产品售前方案助手\n选择一个合成客户案例，查看方案、风险和证据。")
        case_id = gr.Dropdown(choices=list(cases), value="case-001", label="客户案例")
        run = gr.Button("生成方案", variant="primary")
        markdown = gr.Markdown()
        raw = gr.JSON(label="结构化响应")
        run.click(analyze, inputs=case_id, outputs=[markdown, raw])
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "dify"), default="mock")
    parser.add_argument(
        "--share", action="store_true", help="Ask Gradio to create a temporary share link"
    )
    args = parser.parse_args()
    build_app(args.mode).launch(share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
