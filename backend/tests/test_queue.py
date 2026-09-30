"""Queue producer: what gets published to RabbitMQ (pika faked -- no broker needed)."""

import base64
import json

import pytest

from app import queue
from app.config import PRODUCT_IMPORT_QUEUE_NAME


class FakeChannel:
    def __init__(self, publish_error: Exception | None = None) -> None:
        self.publish_error = publish_error
        self.declared: list[dict] = []
        self.published: list[dict] = []

    def queue_declare(self, **kwargs) -> None:
        self.declared.append(kwargs)

    def basic_publish(self, **kwargs) -> None:
        if self.publish_error:
            raise self.publish_error
        self.published.append(kwargs)


class FakeConnection:
    def __init__(self, channel: FakeChannel) -> None:
        self._channel = channel
        self.closed = False

    def channel(self) -> FakeChannel:
        return self._channel

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def broker(monkeypatch):
    channel = FakeChannel()
    connection = FakeConnection(channel)
    monkeypatch.setattr(queue.pika, "BlockingConnection", lambda params: connection)
    return type("Broker", (), {"channel": channel, "connection": connection})()


def _enqueue(**overrides):
    kwargs = dict(
        batch_id="batch1",
        item_index=3,
        sku="shoe-x",
        name="Shoe X",
        price=49.9,
        category="Shoes",
        color="blue",
        image_bytes=b"\x00\x01binary\xff",
        image_filename="image.png",
    )
    kwargs.update(overrides)
    queue.enqueue_import_job(**kwargs)


def test_publishes_one_persistent_message_to_the_durable_import_queue(broker):
    _enqueue()

    assert broker.channel.declared == [{"queue": PRODUCT_IMPORT_QUEUE_NAME, "durable": True}]
    (published,) = broker.channel.published
    assert published["exchange"] == ""
    assert published["routing_key"] == PRODUCT_IMPORT_QUEUE_NAME
    assert published["properties"].delivery_mode == 2  # survives a broker restart
    assert published["properties"].content_type == "application/json"


def test_message_body_carries_metadata_and_base64_image(broker):
    _enqueue()

    body = json.loads(broker.channel.published[0]["body"])
    assert body["batch_id"] == "batch1"
    assert body["item_index"] == 3
    assert (body["sku"], body["name"], body["price"], body["category"], body["color"]) == (
        "shoe-x", "Shoe X", 49.9, "Shoes", "blue",
    )
    assert body["image_filename"] == "image.png"
    # AMQP bodies are bytes; the image survives the JSON round trip exactly.
    assert base64.b64decode(body["image_b64"]) == b"\x00\x01binary\xff"


def test_optional_color_is_sent_as_null(broker):
    _enqueue(color=None)
    assert json.loads(broker.channel.published[0]["body"])["color"] is None


def test_connection_is_closed_after_a_successful_publish(broker):
    _enqueue()
    assert broker.connection.closed


def test_connection_is_closed_even_if_publishing_fails(broker):
    broker.channel.publish_error = RuntimeError("broker went away")

    with pytest.raises(RuntimeError, match="broker went away"):
        _enqueue()

    assert broker.connection.closed
