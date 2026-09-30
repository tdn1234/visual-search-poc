"""AttributeClassifierService: which technique handles category vs. color, and the guards around them."""

from PIL import Image

from app.services.attribute_classifier_service import AttributeClassifierService
from tests.fakes import FakeClip

IMAGE = Image.new("RGB", (4, 4))


class FakeCategoryClassifier:
    def __init__(self, prediction: str) -> None:
        self.prediction = prediction
        self.calls: list[list[float]] = []

    def predict(self, image_embedding: list[float]) -> str:
        self.calls.append(image_embedding)
        return self.prediction


def test_no_labels_means_no_work_and_no_clip_call():
    clip = FakeClip()
    result = AttributeClassifierService(clip).classify(IMAGE, category_labels=None, color_labels=[])

    assert result == (None, None)
    assert clip.encode_calls == 0


def test_image_is_encoded_only_once_when_both_attributes_requested():
    clip = FakeClip(classify_result="red")
    classifier = FakeCategoryClassifier("Shoes")

    AttributeClassifierService(clip, classifier).classify(
        IMAGE, category_labels=["Bags", "Shoes"], color_labels=["red", "blue"]
    )

    assert clip.encode_calls == 1


def test_category_uses_trained_classifier_and_color_uses_zero_shot():
    clip = FakeClip(embedding=[0.2, 0.8], classify_result="blue")
    classifier = FakeCategoryClassifier("Shoes")

    category, color = AttributeClassifierService(clip, classifier).classify(
        IMAGE, category_labels=["Bags", "Shoes"], color_labels=["red", "blue"]
    )

    assert (category, color) == ("Shoes", "blue")
    assert classifier.calls == [[0.2, 0.8]]
    # CLIP was asked only about color, with one prompt per color label.
    assert len(clip.classify_calls) == 1
    assert clip.classify_calls[0][1] == {
        "red": "a photo of something red colored",
        "blue": "a photo of something blue colored",
    }


def test_trained_prediction_outside_current_catalog_categories_is_dropped():
    # e.g. the classifier was trained when a "Hats" category existed, but it's gone now.
    service = AttributeClassifierService(FakeClip(), FakeCategoryClassifier("Hats"))
    category, color = service.classify(IMAGE, category_labels=["Bags", "Shoes"])

    assert category is None
    assert color is None


def test_category_falls_back_to_zero_shot_when_no_classifier_is_loaded(caplog):
    clip = FakeClip(classify_result="Shoes")

    with caplog.at_level("WARNING"):
        category, _ = AttributeClassifierService(clip, None).classify(IMAGE, category_labels=["Bags", "Shoes"])

    assert category == "Shoes"
    assert clip.classify_calls[0][1] == {"Bags": "a photo of bags", "Shoes": "a photo of shoes"}
    assert "no trained category classifier" in caplog.text


def test_color_only_request_skips_category_entirely():
    clip = FakeClip(classify_result="red")
    classifier = FakeCategoryClassifier("Shoes")

    category, color = AttributeClassifierService(clip, classifier).classify(IMAGE, color_labels=["red"])

    assert (category, color) == (None, "red")
    assert classifier.calls == []
