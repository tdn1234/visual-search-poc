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


# --- file logging -----------------------------------------------------------


def _file_handlers():
    import logging
    import logging.handlers

    return [h for h in logging.getLogger().handlers if isinstance(h, logging.handlers.RotatingFileHandler)]


def test_configure_logging_writes_to_a_rotating_file_with_request_ids(tmp_path, monkeypatch):
    import logging

    from app import logging_config

    log_file = tmp_path / "nested" / "app.log"
    monkeypatch.setattr(logging_config, "LOG_FILE", str(log_file))
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers, root.level
    try:
        logging_config.configure_logging()
        token = logging_config.request_id_var.set("req123")
        logging.getLogger("app.test").warning("hello file")
        logging_config.request_id_var.reset(token)
        for handler in _file_handlers():
            handler.flush()
        assert "[WARNING] [req123] app.test: hello file" in log_file.read_text()
    finally:
        for handler in _file_handlers():
            handler.close()
        root.handlers, root.level = saved_handlers, saved_level


def test_empty_log_file_setting_disables_file_logging(monkeypatch):
    import logging

    from app import logging_config

    monkeypatch.setattr(logging_config, "LOG_FILE", "")
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers, root.level
    try:
        logging_config.configure_logging()
        assert _file_handlers() == []
    finally:
        root.handlers, root.level = saved_handlers, saved_level


def test_unwritable_log_location_falls_back_to_console_only(tmp_path, monkeypatch):
    import logging

    from app import logging_config

    blocker = tmp_path / "file"
    blocker.write_text("x")  # a *file* where a directory is needed
    monkeypatch.setattr(logging_config, "LOG_FILE", str(blocker / "app.log"))
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers, root.level
    try:
        logging_config.configure_logging()
        assert _file_handlers() == []
        assert root.handlers  # console handler still installed
    finally:
        root.handlers, root.level = saved_handlers, saved_level
