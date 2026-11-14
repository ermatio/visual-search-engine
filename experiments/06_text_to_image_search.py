from pathlib import Path
import torch
from visual_search import ClipEmbedder

image_directory = Path("data/images")
supported_extensions = {".jpg", ".jpeg", ".png"}

image_paths = sorted(
    path
    for path in image_directory.iterdir()
    if path.suffix.lower() in supported_extensions
)

if not image_paths:
    raise ValueError("image not found")

embedder = ClipEmbedder()

image_embeddings = embedder.encode_images(image_paths)

query = input("Describe the image you want: ")
text_embedding = embedder.encode_text(query)

similarity_scores = image_embeddings @ text_embedding.T
similarity_scores = similarity_scores.squeeze(dim=1)

scores, indexes = torch.sort(similarity_scores, descending = True)

print(f"\nResults for: {query}\n")

for score, index in zip(scores, indexes):
    print(f"{image_paths[index].name} : {score.item():.3f}")