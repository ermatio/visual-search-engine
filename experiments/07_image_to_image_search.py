from pathlib import Path
import open_clip
import torch
from PIL import Image

model_name = "ViT-B-32-quickgelu"
dataset_directory = Path("data/images")
supported_extensions = {".jpg", ".jpeg", ".png"}

image_paths = sorted(
    path
    for path in dataset_directory.iterdir()
    if path.suffix.lower() in supported_extensions
)

if not image_paths:
    raise ValueError("dataset not found")

query_path = Path(
    input("Enter the query image path: ").strip().strip('"')
)

if not query_path.exists():
    raise FileNotFoundError(f"Query image not found: {query_path}")

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained= "openai"
)

model.eval()

processed_images = []

for path in image_paths:
    with Image.open(path) as image:
        processed_images.append(preprocess(image.convert("RGB")))

image_batch = torch.stack(processed_images)

with Image.open(query_path) as image:
    query_tensor = preprocess(image.convert("RGB")).unsqueeze(0)

with torch.inference_mode():
    image_embeddings = model.encode_image(image_batch)
    query_embedding = model.encode_image(query_tensor)
    image_embeddings = image_embeddings / image_embeddings.norm(
        dim = -1,
        keepdim = True
    )
    query_embedding = query_embedding / query_embedding.norm(
        dim = -1,
        keepdim = True
    )

similarity_score = image_embeddings @ query_embedding.T
similarity_score = similarity_score.squeeze(1)

scores, indexes = torch.sort(similarity_score, descending = True)

for score, index in zip(scores, indexes):
    print(f"{image_paths[index].name} : {score.item():.3f}")