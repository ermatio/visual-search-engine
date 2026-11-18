from pathlib import Path
import pytest
import torch
from visual_search import ExactImageIndex

def create_test_index(temporary_directory: Path) -> Path:
    index_file_path = temporary_directory / "image_index.pt"

    saved_index = {
        "model_name": "test-model",
        "pretrained": "test-weights",
        "image_paths": [
            "data/images/dog.jpg",
            "data/images/cat.jpg",
            "data/images/car.jpg",
        ],
        "embeddings": torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]),
    }

    torch.save(saved_index, index_file_path)

    return index_file_path


def test_search_returns_most_similar_image_first(
    tmp_path: Path,
) -> None:
    index_file_path = create_test_index(tmp_path)

    exact_index = ExactImageIndex(
        index_file_path=index_file_path,
        project_root_path=tmp_path,
    )

    query_embedding = torch.tensor([[1.0, 0.0, 0.0]])

    search_results = exact_index.search(
        query_embedding=query_embedding,
        top_k=2,
    )

    assert search_results[0][0].name == "dog.jpg"
    assert search_results[0][1] == pytest.approx(1.0)
    assert len(search_results) == 2


def test_search_rejects_invalid_query_shape(
    tmp_path: Path,
) -> None:
    index_file_path = create_test_index(tmp_path)

    exact_index = ExactImageIndex(
        index_file_path=index_file_path,
        project_root_path=tmp_path,
    )

    invalid_query = torch.tensor([1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="shape"):
        exact_index.search(invalid_query)


def test_search_rejects_invalid_top_k(
    tmp_path: Path,
) -> None:
    index_file_path = create_test_index(tmp_path)

    exact_index = ExactImageIndex(
        index_file_path=index_file_path,
        project_root_path=tmp_path,
    )

    query_embedding = torch.tensor([[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="greater than zero"):
        exact_index.search(query_embedding, top_k=0)