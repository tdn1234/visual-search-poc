"""POST /agent/chat: form validation, image handoff, session memory, error mapping."""

import httpx

from app.config import AGENT_MESSAGE_MAX_LENGTH


def _chat(client, headers, data=None, files=None):
    return client.post("/agent/chat", headers=headers, data=data or {"message": "red shoes"}, files=files)


def test_requires_api_key(client):
    assert _chat(client, {}).status_code == 401


def test_returns_reply_products_session_and_trace(client, auth_headers, services):
    response = _chat(client, auth_headers)

    body = response.json()
    assert response.status_code == 200
    assert body["reply"] == "Here you go."
    assert body["products"] == [
        {"sku": "shoe-red", "name": "Red Shoe", "price": 99.0, "category": "Shoes", "score": 0.9}
    ]
    assert len(body["session_id"]) == 32
    assert body["steps"][0]["tool"] == "get_product"


def test_shopper_id_and_message_reach_the_agent(client, auth_headers, services):
    _chat(client, auth_headers, data={"message": "hi", "shopper_id": "shopper-1"})

    assert services.agent.chat_calls == [{"message": "hi", "history": [], "shopper_id": "shopper-1"}]


def test_history_is_saved_and_reused_for_the_same_session(client, auth_headers, services):
    first = _chat(client, auth_headers, data={"message": "one"}).json()
    _chat(client, auth_headers, data={"message": "two", "session_id": first["session_id"]})

    assert services.agent.chat_calls[1]["history"] == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "Here you go."},
    ]
    assert len(services.sessions.sessions[first["session_id"]]) == 4


def test_uploaded_image_is_stored_and_only_its_ref_goes_to_the_model(client, auth_headers, services, png_bytes):
    _chat(client, auth_headers, files={"file": ("a.png", png_bytes, "image/png")})

    assert services.agent.images == [png_bytes]
    message = services.agent.chat_calls[0]["message"]
    assert "image_ref=img_1" in message and message.endswith("red shoes")


def test_bad_image_is_rejected(client, auth_headers, services):
    response = _chat(client, auth_headers, files={"file": ("a.png", b"nope", "image/png")})

    assert response.status_code == 400
    assert services.agent.chat_calls == []


def test_invalid_form_fields_are_422(client, auth_headers):
    assert _chat(client, auth_headers, data={"message": ""}).status_code == 422
    assert _chat(client, auth_headers, data={"message": "x" * (AGENT_MESSAGE_MAX_LENGTH + 1)}).status_code == 422
    assert _chat(client, auth_headers, data={"message": "x", "session_id": "bad id!"}).status_code == 422
    assert _chat(client, auth_headers, data={"message": "x", "shopper_id": "bad id!"}).status_code == 422


def test_llm_unreachable_is_503(client, auth_headers, services):
    services.agent.error = httpx.ConnectError("down")

    assert _chat(client, auth_headers).status_code == 503


def test_unexpected_error_is_500_without_leaking_details(client, auth_headers, services):
    services.agent.error = RuntimeError("secret internals")

    response = _chat(client, auth_headers)
    assert response.status_code == 500
    assert "secret" not in response.text


def test_session_store_failure_does_not_break_chat(client, auth_headers, services):
    services.sessions.error = ConnectionError("redis down")

    assert _chat(client, auth_headers).status_code == 200


def test_rate_limited(client, auth_headers):
    codes = [_chat(client, auth_headers).status_code for _ in range(4)]

    assert codes == [200, 200, 200, 429]


def test_swagger_documents_the_endpoint_with_api_key_auth(client):
    operation = client.get("/openapi.json").json()["paths"]["/agent/chat"]["post"]

    assert operation["tags"] == ["agent"]
    assert operation["security"] == [{"APIKeyHeader": []}]
    assert "multipart/form-data" in operation["requestBody"]["content"]
