from pathlib import Path
import torch
from visual_search import ClipEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIRECTORY = PROJECT_ROOT / "data" / "images"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "image_index.pt"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

image_paths = sorted(
    path 
    for path in IMAGE_DIRECTORY.iterdir()
    if path.suffix.lower() in SUPPORTED_EXTENSIONS
)

if not image_paths:
    raise ValueError("No supported images found")

embedding_model = ClipEmbedder()
image_embeddings = embedding_model.encode_images(image_paths)

relative_image_paths = [
    image_path.relative_to(PROJECT_ROOT).as_posix()
    for image_path in image_paths 
]

index_data = {
    "model_name": embedding_model.model_name,
    "pretrained": embedding_model.pretrained,
    "image_paths": relative_image_paths,
    "embeddings": image_embeddings.cpu()
}

INDEX_PATH.parent.mkdir(parents = True, exist_ok = True)
torch.save(index_data,INDEX_PATH)

print(f"Indexed {len(image_paths)} images")
print(f"Embedding shape: {image_embeddings.shape}")
print(f"Index saved to: {INDEX_PATH}")