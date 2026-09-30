"""EmbeddingService: image validation + optional color adapter, with CLIP faked."""

import pytest
import torch
from PIL import Image

from app.models.color_adapter import ColorAdapter
from app.services.embedding_service import EmbeddingService
from tests.fakes import FakeClip


def test_embed_image_bytes_returns_clip_embedding_when_no_adapter(png_bytes):
    clip = FakeClip(embedding=[0.3, 0.4])
    assert EmbeddingService(clip).embed_image_bytes(png_bytes) == [0.3, 0.4]
    assert clip.encode_calls == 1


def test_embed_image_bytes_rejects_non_image_bytes():
    clip = FakeClip()
    with pytest.raises(ValueError, match="not a valid image"):
        EmbeddingService(clip).embed_image_bytes(b"definitely not an image")
    assert clip.encode_calls == 0  # never reached CLIP


def test_embed_image_file_reads_from_disk(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(path)
    clip = FakeClip(embedding=[0.1, 0.9])

    assert EmbeddingService(clip).embed_image_file(path) == [0.1, 0.9]


def test_embed_image_file_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        EmbeddingService(FakeClip()).embed_image_file(tmp_path / "nope.png")


def test_embed_image_file_rejects_a_file_that_is_not_an_image(tmp_path):
    path = tmp_path / "image.png"
    path.write_text("not an image")
    with pytest.raises(ValueError, match="not a valid image"):
        EmbeddingService(FakeClip()).embed_image_file(path)


def test_color_adapter_is_applied_to_every_embedding(png_bytes, tmp_path):
    adapter = ColorAdapter(embedding_dim=2, hidden_dim=2)
    for param in adapter.parameters():
        torch.nn.init.zeros_(param)  # residual head -> output is just the normalized input
    service = EmbeddingService(FakeClip(embedding=[3.0, 4.0]), color_adapter=adapter)

    from_bytes = service.embed_image_bytes(png_bytes)
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    from_file = service.embed_image_file(path)

    # Catalog (file) and query (bytes) must go through the same transformation.
    assert from_bytes == pytest.approx([0.6, 0.8], abs=1e-6)
    assert from_file == pytest.approx(from_bytes)
