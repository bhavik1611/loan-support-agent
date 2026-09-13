"""Task 11. The FastAPI deployment."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_36_ask_answers_a_policy_question_through_the_api():
    """Spec section 16 test 36, Part 3 criterion 1."""
    response = client.post("/ask", json={"query": "How is the EMI on a loan calculated?"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "policy"
    assert body["answer"]
    assert body["policy"]["citations"]


def test_ask_routes_a_lookup_question_to_the_record_tool():
    response = client.post("/ask", json={"query": "What is the status of LN-1042?"})
    assert response.status_code == 200
    assert response.json()["route"] == "lookup"


def test_ask_defaults_the_thread_id():
    response = client.post("/ask", json={"query": "How is the EMI calculated?"})
    assert response.json()["thread_id"] == "default"


def test_ask_rejects_an_empty_query_with_422():
    assert client.post("/ask", json={"query": ""}).status_code == 422


def test_the_response_matches_the_committed_schema():
    """The envelope is AgentResponse unchanged, per D-66. Nothing is widened."""
    import json

    import config

    schema = json.loads(config.RESPONSE_SCHEMA_PATH.read_text())
    body = client.post("/ask", json={"query": "How is the EMI calculated?"}).json()
    import jsonschema

    jsonschema.validate(body, schema)
