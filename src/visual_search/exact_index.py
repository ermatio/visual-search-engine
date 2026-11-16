from pathlib import Path
import torch
from torch import Tensor

class ExactImageIndex:
    """Load saved image embeddings and perform exact similarity search"""

    def __init__(
            self,
            index_file_path: Path,
            project_root_path: Path,
    ) -> None:
        if not index_file_path.exists():
            raise FileNotFoundError(
                f"Index file not found: {index_file_path}"
            )

        saved_index = torch.load(
            index_file_path,
            map_location = "cpu",
            weights_only = True,
        )

        self.model_name = saved_index["model_name"]
        self.pretrained = saved_index["pretrained"]

        self.image_paths = [
            project_root_path / relative_path
            for relative_path in saved_index["image_paths"]
        ]

        self.image_embeddings = saved_index["embeddings"]

        if len(self.image_paths) != len(self.image_embeddings):
            raise ValueError("Image paths and embeddings have different lengths")

    def search(
            self,
            query_embedding: Tensor,
            top_k: int = 5,
    ) -> list[tuple[Path, float]] :

        if query_embedding.ndim !=2 or query_embedding.shape[0] != 1:
            raise ValueError("Query embedding must have shape [1, dimensions]")

        if query_embedding.shape[1] != self.image_embeddings.shape[1]:
            raise ValueError("Query and image embedding dimensions do not match")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

            # start
        result_count = min(top_k, self.image_embeddings.shape[0])

        similarity_scores = (query_embedding.cpu() @ self.image_embeddings.T).squeeze(dim = 0)

        top_scores, indexes = torch.topk(
            similarity_scores,
            k = result_count,
        )

        results = []

        for score, index in zip(top_scores, indexes):
            results.append( (self.image_paths[index], score))

        return results