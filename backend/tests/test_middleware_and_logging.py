"""Request-ID correlation (middleware + logging filter) and the unauthenticated health check."""

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.logging_config import _RequestIdFilter, request_id_var
from app.middleware import RequestContextMiddleware


def _mini_app() -> FastAPI:
    """Just the middleware + a route that reveals the request ID its handler sees."""
    mini = FastAPI()
    mini.add_middleware(RequestContextMiddleware)

    @mini.get("/whoami")
    async def whoami():
        return {"request_id": request_id_var.get()}

    @mini.get("/boom")
    async def boom():
        raise RuntimeError("kaboom")

    return mini


def test_generates_a_request_id_visible_to_handlers_and_echoed_in_the_response():
    response = TestClient(_mini_app()).get("/whoami")

    generated = response.headers["X-Request-ID"]
    assert len(generated) == 12
    assert response.json() == {"request_id": generated}


def test_trusts_an_inbound_request_id_so_ids_thread_across_services():
    response = TestClient(_mini_app()).get("/whoami", headers={"X-Request-ID": "from-proxy-123"})

    assert response.headers["X-Request-ID"] == "from-proxy-123"
    assert response.json() == {"request_id": "from-proxy-123"}


def test_each_request_gets_a_distinct_id():
    client = TestClient(_mini_app())
    assert client.get("/whoami").headers["X-Request-ID"] != client.get("/whoami").headers["X-Request-ID"]


def test_logs_one_access_line_with_method_path_and_status(caplog):
    with caplog.at_level(logging.INFO, logger="app.middleware"):
        TestClient(_mini_app()).get("/whoami")

    (line,) = [r.getMessage() for r in caplog.records if r.name == "app.middleware"]
    assert line.startswith("GET /whoami -> 200 (")


def test_logs_unhandled_exceptions_and_still_propagates_them(caplog):
    client = TestClient(_mini_app(), raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="app.middleware"):
        response = client.get("/boom")

    assert response.status_code == 500
    assert "GET /boom -> unhandled exception" in caplog.text


def test_log_filter_stamps_the_current_request_id_on_records():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)

    token = request_id_var.set("abc123")
    try:
        assert _RequestIdFilter().filter(record) is True
        assert record.request_id == "abc123"
    finally:
        request_id_var.reset(token)


def test_log_filter_uses_empty_string_outside_a_request():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
    _RequestIdFilter().filter(record)
    assert record.request_id == ""


def test_health_check_needs_no_api_key(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
