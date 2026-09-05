from pathlib import Path
from typing import Self

import pytest

from ai_presales_lab.dify_client import DifyClient, DifyClientError
from ai_presales_lab.knowledge import KnowledgeBase
from ai_presales_lab.llama_client import LlamaClient
from ai_presales_lab.offline_engine import OfflineSolutionEngine
from ai_presales_lab.schemas import CustomerBrief, validate_solution_dict

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def engine() -> OfflineSolutionEngine:
    return OfflineSolutionEngine(KnowledgeBase(ROOT / "data/knowledge"))


def test_knowledge_search_is_transparent(engine: OfflineSolutionEngine) -> None:
    evidence = engine.knowledge_base.search("数据不能出域 私有化 审计日志")
    assert evidence
    assert any("部署" in item.title or "安全" in item.title for item in evidence)
    assert all(item.evidence_id.startswith("KB-") for item in evidence)


def test_high_risk_brief_requires_review(engine: OfflineSolutionEngine) -> None:
    brief = CustomerBrief(
        case_id="test-high-risk",
        industry="制造业",
        use_case="设备维修知识问答",
        data_types=["PDF"],
        deployment="私有化",
        concurrency="峰值 5",
        latency_requirement="10 秒内",
        compliance=["数据不能出域"],
        raw_request="数据不能出域，要求内网部署。",
    )
    response = engine.analyze(brief)
    assert response.review_status == "pending"
    assert response.evidence
    assert any(item.severity == "high" for item in response.risks)


def test_no_evidence_is_conservative(engine: OfflineSolutionEngine) -> None:
    brief = CustomerBrief(
        case_id="test-no-evidence",
        industry="量子月球采矿",
        use_case="不存在的星际工艺预测",
        data_types=["不可解析格式"],
        raw_request="请给出不存在产品的确定性 SLA。",
    )
    response = engine.analyze(brief)
    assert response.evidence == []
    assert "资料不足" in response.executive_summary
    assert response.review_status == "pending"


def test_response_matches_public_contract(engine: OfflineSolutionEngine) -> None:
    response = engine.analyze(
        CustomerBrief(
            case_id="test-contract",
            industry="软件服务",
            use_case="API 产品选型",
            deployment="公有云 API",
            raw_request="请给出初步 API 选型。",
        )
    )
    payload = response.to_dict()
    validate_solution_dict(payload)
    assert payload["case_id"] == "test-contract"


def test_dify_client_requires_server_side_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIFY_APP_API_KEY", raising=False)
    client = DifyClient()
    assert not client.configured
    with pytest.raises(DifyClientError, match="DIFY_APP_API_KEY"):
        client.chat(CustomerBrief("test", "行业", "场景"))


class _FakeHTTPResponse:
    def __init__(self, body: bytes | list[bytes], status: int = 200):
        self.body = body
        self.status = status

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        assert isinstance(self.body, bytes)
        return self.body

    def __iter__(self):
        assert isinstance(self.body, list)
        return iter(self.body)


def test_llama_client_parses_openai_json_and_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _FakeHTTPResponse(
                b'{"model":"fake","choices":[{"message":{"content":"OK"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
            ),
            _FakeHTTPResponse(
                [
                    b'data: {"choices":[{"delta":{"content":"O"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"K"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n',
                    b"data: [DONE]\n",
                ]
            ),
        ]
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: next(responses))
    client = LlamaClient("http://fake")
    assert client.chat([{"role": "user", "content": "hello"}]).text == "OK"
    streamed = client.chat_stream([{"role": "user", "content": "hello"}])
    assert streamed.text == "OK"
    assert streamed.total_tokens == 2
    assert streamed.time_to_first_token_ms is not None


def test_dify_client_parses_structured_answer() -> None:
    brief = CustomerBrief("case", "行业", "场景")
    payload = {
        "answer": '{"executive_summary":"有依据的方案","recommendation":[],"architecture":[],"implementation_steps":[],"risks":[],"clarifying_questions":[],"evidence":[],"review_status":"not_required"}'
    }
    response = DifyClient._to_solution(payload, brief)
    assert response.case_id == "case"
    assert response.executive_summary == "有依据的方案"
