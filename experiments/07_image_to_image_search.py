from pathlib import Path 
import torch
from visual_search import ClipEmbedder

project_root = Path(__file__).resolve().parents[1]
dataset_directory = project_root/"data"/"images"
supported_extensions = {".jpg", ".jpeg", ".png"}

image_paths = sorted(
    path
    for path in dataset_directory.iterdir()
    if path.suffix.lower() in supported_extensions
)

if not image_paths:
    raise ValueError("No supported dataset images found")

query_input = Path(
    input("Enter the query image path: ").strip().strip('"')
)

if query_input.is_absolute():
    query_path = query_input
else:
    query_path = project_root/query_input

if not query_path.exists():
    raise FileNotFoundError(f"Query image not found: {query_path}")


embedder = ClipEmbedder()

image_embeddings = embedder.encode_images(image_paths)
query_embedding = embedder.encode_images([query_path])

similarity_scores = image_embeddings @ query_embedding.T
similarity_scores = similarity_scores.squeeze(dim=1)

scores, indexes = torch.sort(similarity_scores, descending=True)

print(f"\nImages similar to: {query_path.name}\n")

for similarity_score, index in zip(scores, indexes):
    print(f"{image_paths[index].name} : {similarity_score.item():.3f}")    