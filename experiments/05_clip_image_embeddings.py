from pathlib import Path
import open_clip
from PIL import Image
import torch

image_directory = Path("data/images")
supported_extensions = {".jpg", ".jpeg", ".png"}

image_paths = sorted(
    path
    for path in image_directory.iterdir()
    if path.suffix.lower() in supported_extensions
    )

if not image_paths:
    raise ValueError("no supported file found")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu",
    pretrained = "openai"
)

model.eval()

processed_images = []

for image_path in image_paths:
    with Image.open(image_path) as image:
        processed_image = preprocess(image.convert("RGB"))
        processed_images.append(processed_image)

image_batch = torch.stack(processed_images)

with torch.inference_mode():
    image_embeddings = model.encode_image(image_batch)
    image_embeddings = image_embeddings / image_embeddings.norm(
        dim = -1,
        keepdim = True
    )

print("batch shape:", image_batch.shape)
print("image embeddings shape:", image_embeddings.shape)

for image, embedding in zip(image_paths, image_embeddings):
    print(f"{image.name}: {embedding[:5]}")