"""Color adapter and category classifier heads (small torch modules, no CLIP needed)."""

import pytest
import torch

from app.models.category_classifier import CategoryClassifier, load_category_classifier
from app.models.color_adapter import ColorAdapter, load_color_adapter


# --- ColorAdapter -----------------------------------------------------------


def _zeroed_adapter(dim: int = 2, hidden: int = 4) -> ColorAdapter:
    adapter = ColorAdapter(embedding_dim=dim, hidden_dim=hidden)
    for param in adapter.parameters():
        torch.nn.init.zeros_(param)
    return adapter


def test_adapter_output_is_l2_normalized():
    adapter = ColorAdapter(embedding_dim=16, hidden_dim=8)
    out = adapter(torch.randn(5, 16))
    assert out.shape == (5, 16)
    assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_adapter_is_residual_so_a_zeroed_head_only_normalizes():
    # net(x) == 0  ->  x + 0, then re-normalized: proves the residual connection.
    out = _zeroed_adapter()(torch.tensor([[3.0, 4.0]]))
    assert torch.allclose(out, torch.tensor([[0.6, 0.8]]), atol=1e-6)


def test_load_color_adapter_returns_none_when_no_checkpoint(tmp_path):
    assert load_color_adapter(tmp_path / "missing.pt") is None


def test_load_color_adapter_round_trips_weights_and_sets_eval_mode(tmp_path):
    original = ColorAdapter(embedding_dim=8, hidden_dim=4)
    checkpoint = tmp_path / "adapter.pt"
    torch.save({"embedding_dim": 8, "hidden_dim": 4, "state_dict": original.state_dict()}, checkpoint)

    loaded = load_color_adapter(checkpoint)

    assert loaded is not None
    assert not loaded.training
    sample = torch.randn(3, 8)
    assert torch.allclose(loaded(sample), original(sample))


# --- CategoryClassifier -----------------------------------------------------


def _identity_classifier(labels: list[str]) -> CategoryClassifier:
    """Logit i == embedding[i], so predict() returns the label of the largest component."""
    classifier = CategoryClassifier(embedding_dim=len(labels), category_labels=labels)
    with torch.no_grad():
        classifier.linear.weight.copy_(torch.eye(len(labels)))
        classifier.linear.bias.zero_()
    return classifier


@pytest.mark.parametrize(
    ("embedding", "expected"),
    [
        ([0.9, 0.1, 0.0], "Bags"),
        ([0.1, 0.8, 0.1], "Hats"),
        ([0.0, 0.2, 0.7], "Shoes"),
    ],
)
def test_predict_returns_the_highest_scoring_label(embedding, expected):
    assert _identity_classifier(["Bags", "Hats", "Shoes"]).predict(embedding) == expected


def test_forward_returns_one_logit_per_label():
    classifier = CategoryClassifier(embedding_dim=6, category_labels=["A", "B", "C", "D"])
    assert classifier(torch.randn(2, 6)).shape == (2, 4)


def test_load_category_classifier_returns_none_when_no_checkpoint(tmp_path):
    assert load_category_classifier(tmp_path / "missing.pt") is None


def test_load_category_classifier_round_trips_labels_and_weights(tmp_path):
    original = _identity_classifier(["Bags", "Hats", "Shoes"])
    checkpoint = tmp_path / "classifier.pt"
    torch.save(
        {
            "embedding_dim": 3,
            "category_labels": original.category_labels,
            "state_dict": original.state_dict(),
        },
        checkpoint,
    )

    loaded = load_category_classifier(checkpoint)

    assert loaded is not None
    assert not loaded.training
    assert loaded.category_labels == ["Bags", "Hats", "Shoes"]
    assert loaded.predict([0.0, 0.1, 0.9]) == "Shoes"
