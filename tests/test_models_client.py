import io
import json
from urllib.error import URLError

import pytest

from src.llm_client import LLMResponse, LocalClient, LocalConfig, MockClient, NoRedirect, create_client
from src.models import AdaptiveAgent, ICDBaseline, MultiAgent, SingleLLM
from src.schemas import DiagnosisCode, Patient, Usage
from src.synthetic import SYNTHETIC_POSITIVE


class ScriptedClient:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = []

    def score(self, patient, role):
        self.calls.append((patient, role))
        assert not hasattr(patient, "label")
        return LLMResponse(next(self.scores), Usage(10, 2, 1))

    def describe(self):
        return {"backend": "synthetic-test"}


def test_baseline_exact_system_and_code_matching():
    model = ICDBaseline(frozenset({SYNTHETIC_POSITIVE}), code_set_version="synthetic-test")
    assert model.predict(Patient("synthetic-p", (SYNTHETIC_POSITIVE,))).predicted_label == 1
    assert model.predict(Patient("synthetic-p", (DiagnosisCode("OTHER", SYNTHETIC_POSITIVE.code),))).score == 0
    assert model.predict(Patient("synthetic-p")).usage == Usage()
    with pytest.raises(ValueError):
        ICDBaseline(frozenset(), code_set_version="missing")


def test_single_and_multi_use_diagnoses_and_text_and_count_costs():
    patient = Patient("synthetic-p", (SYNTHETIC_POSITIVE,), "invented text")
    client = ScriptedClient([0.5, 0.2, 0.8])
    single = SingleLLM(client).predict(patient)
    multi = MultiAgent(client).predict(patient)
    assert single.predicted_label == 1
    assert multi.score == pytest.approx(0.5)
    assert multi.usage == Usage(20, 4, 2)
    assert all(p is patient for p, _ in client.calls)
    assert len({role for _, role in client.calls}) == 3


@pytest.mark.parametrize("initial,escalates", [(0.1, False), (0.35, True), (0.5, True), (0.65, True), (0.9, False)])
def test_adaptive_routes_and_inclusive_boundaries(initial, escalates):
    cheap, expensive = ScriptedClient([initial]), ScriptedClient([0.8, 0.9])
    prediction = AdaptiveAgent(cheap, expensive_client=expensive).predict(Patient("synthetic-p"))
    assert prediction.usage.llm_calls == (3 if escalates else 1)
    assert prediction.usage.input_tokens == (30 if escalates else 10)
    assert prediction.score == pytest.approx(0.85 if escalates else initial)
    assert prediction.route == ("single_to_multi" if escalates else "single_only")
    assert prediction.latency_seconds >= 0


@pytest.mark.parametrize("kwargs", [{"lower": 0.7}, {"upper": 0.2}, {"lower": 0.5, "upper": 0.5}, {"threshold": 2}])
def test_bad_adaptive_configuration(kwargs):
    with pytest.raises(ValueError):
        AdaptiveAgent(MockClient(), **kwargs)


def test_mock_is_deterministic_and_factory_validates_backend():
    patient = Patient("synthetic-p")
    assert create_client().score(patient, "test") == MockClient().score(patient, "test")
    with pytest.raises(ValueError):
        create_client("remote")
    with pytest.raises(ValueError):
        create_client("local")


@pytest.mark.parametrize("url", ["https://example.com", "http://example.com", "http://127.0.0.1.evil", "http://user@127.0.0.1", "http://127.0.0.1/path", "http://127.0.0.1?x=1"])
def test_non_loopback_and_ambiguous_endpoints_rejected(url):
    with pytest.raises(ValueError):
        LocalConfig("synthetic-model", approved=True, base_url=url)


def test_approval_and_redirect_guards():
    with pytest.raises(ValueError, match="approval"):
        LocalConfig("synthetic-model")
    with pytest.raises(ValueError, match="redirect"):
        NoRedirect().redirect_request(None, None, 302, "", {}, "http://example.com")


class FakeOpener:
    def __init__(self, body):
        self.body = body
        self.request = None

    def open(self, request, timeout):
        self.request = request
        if isinstance(self.body, Exception):
            raise self.body
        return io.BytesIO(json.dumps(self.body).encode())


def local_response(content='{"score": 0.7}', usage=None):
    return {"choices": [{"message": {"content": content}}], "usage": usage}


def test_local_serialization_and_unknown_usage_without_network():
    client = LocalClient(LocalConfig("synthetic-model", approved=True))
    opener = FakeOpener(local_response())
    client._opener = opener
    response = client.score(Patient("synthetic-private-id", (SYNTHETIC_POSITIVE,), "invented text"), "test")
    assert response == LLMResponse(0.7, Usage(None, None, 1))
    payload = json.loads(opener.request.data)
    assert "synthetic-private-id" not in opener.request.data.decode()
    assert "DEMO_POSITIVE" in payload["messages"][1]["content"]
    assert "invented text" in payload["messages"][1]["content"]
    assert payload["model"] == "synthetic-model"
    opener.body = local_response(usage={"prompt_tokens": 11, "completion_tokens": 4})
    assert client.score(Patient("synthetic-p"), "test").usage == Usage(11, 4, 1)


@pytest.mark.parametrize("body", [local_response("not JSON"), local_response('{"score": 3}'),
                                 local_response('{"score": true}'), {},
                                 local_response(usage=["invalid-usage"]),
                                 local_response(usage={"prompt_tokens": -1}), URLError("secret-response")])
def test_invalid_responses_fail_without_exposing_response_body(body):
    client = LocalClient(LocalConfig("synthetic-model", approved=True))
    client._opener = FakeOpener(body)
    with pytest.raises(RuntimeError) as error:
        client.score(Patient("synthetic-secret-id"), "test")
    assert "secret" not in str(error.value)
