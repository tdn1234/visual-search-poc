"""Worker-side import job: message decoding and the permanent-vs-unexpected failure split."""

import base64
import json

import pytest

from app import jobs
from app.logging_config import request_id_var

JOB = dict(
    batch_id="batch1",
    item_index=2,
    sku="bag-x",
    name="Bag X",
    price=9.5,
    category="Bags",
    color="red",
    image_bytes=b"\x89PNG-bytes",
    image_filename="image.png",
)


class RecordingIndexing:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict] = []
        self.request_id_during_call: str | None = None

    update_error: Exception | None = None
    update_calls: list = []

    def add_product(self, **kwargs):
        self.calls.append(kwargs)
        self.request_id_during_call = request_id_var.get()
        if self.error:
            raise self.error

    def update_product(self, **kwargs):
        self.update_calls.append(kwargs)
        if self.update_error:
            raise self.update_error


@pytest.fixture
def indexing(monkeypatch):
    fake = RecordingIndexing()
    fake.update_calls = []
    monkeypatch.setattr(jobs, "_indexing_service", fake)
    return fake


def test_job_fails_loudly_if_init_services_was_never_called(monkeypatch):
    monkeypatch.setattr(jobs, "_indexing_service", None)
    with pytest.raises(RuntimeError, match="init_services"):
        jobs.import_product_job(**JOB)


def test_job_calls_add_product_with_the_same_arguments_the_api_uses(indexing):
    jobs.import_product_job(**JOB)

    (call,) = indexing.calls
    assert call["sku"] == "bag-x"
    assert call["image_bytes"] == b"\x89PNG-bytes"
    assert call["image_filename"] == "image.png"
    assert call["color"] == "red"


def test_job_tags_logs_with_batch_and_item_then_restores_the_request_id(indexing):
    jobs.import_product_job(**JOB)

    assert indexing.request_id_during_call == "import-batch1-2"
    assert request_id_var.get() == ""


def test_existing_sku_is_updated_instead_of_skipped(indexing):
    indexing.error = FileExistsError("already exists")

    jobs.import_product_job(**JOB)

    (call,) = indexing.update_calls
    assert call["sku"] == "bag-x"
    assert call["image_bytes"] == b"\x89PNG-bytes"


def test_existing_sku_whose_update_fails_permanently_is_skipped(indexing, caplog):
    indexing.error = FileExistsError("already exists")
    indexing.update_error = ValueError("bad image")

    with caplog.at_level("WARNING"):
        jobs.import_product_job(**JOB)  # must not raise

    assert "Skipped product 'bag-x'" in caplog.text


@pytest.mark.parametrize("permanent_error", [ValueError("bad image")])
def test_permanent_failures_are_swallowed_so_the_message_is_acked(indexing, permanent_error, caplog):
    """A duplicate sku / corrupt image fails identically on every retry -- redelivering would loop forever."""
    indexing.error = permanent_error

    with caplog.at_level("WARNING"):
        jobs.import_product_job(**JOB)  # must not raise

    assert "Skipped product 'bag-x'" in caplog.text
    assert "batch 'batch1'" in caplog.text


def test_unexpected_failures_are_reraised_so_the_message_is_redelivered(indexing):
    """Postgres down etc.: re-raising leaves the RabbitMQ message unacked instead of losing the product."""
    indexing.error = ConnectionError("postgres unreachable")

    with pytest.raises(ConnectionError):
        jobs.import_product_job(**JOB)


def test_request_id_is_restored_even_when_the_job_raises(indexing):
    indexing.error = ConnectionError("boom")

    with pytest.raises(ConnectionError):
        jobs.import_product_job(**JOB)

    assert request_id_var.get() == ""


def test_handle_message_decodes_the_wire_format_produced_by_the_queue(indexing):
    body = json.dumps(
        {
            "batch_id": "b9",
            "item_index": 0,
            "sku": "hat-1",
            "name": "Hat",
            "price": 3.0,
            "category": "Hats",
            "color": None,
            "image_filename": "image.jpg",
            "image_b64": base64.b64encode(b"raw-image").decode("ascii"),
        }
    ).encode("utf-8")

    jobs.handle_message(body)

    (call,) = indexing.calls
    assert call["sku"] == "hat-1"
    assert call["color"] is None
    assert call["image_bytes"] == b"raw-image"  # base64 decoded back to bytes
    assert indexing.request_id_during_call == "import-b9-0"


def test_handle_message_propagates_unexpected_errors_so_the_worker_does_not_ack(indexing):
    indexing.error = ConnectionError("boom")
    body = json.dumps(
        {**{k: v for k, v in JOB.items() if k != "image_bytes"}, "image_b64": base64.b64encode(b"x").decode()}
    ).encode()

    with pytest.raises(ConnectionError):
        jobs.handle_message(body)
