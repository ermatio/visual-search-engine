from pathlib import Path
import open_clip
import torch
from PIL import Image

model_name = "ViT-B-32"
image_directory = Path("data/images")
supported_extensions = {".jpg", ".jpeg", ".png"}

image_path = sorted(
    path
    for path in image_directory.iterdir()
    if path.suffix.lower() in supported_extensions
)

if not image_path:
    raise ValueError("image not found")

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained= "openai"    
)

tokenizer = open_clip.get_tokenizer(model_name)
model.eval()

processed_images = []

for path in image_path:
    with Image.open(path) as image:
        processed_images.append(preprocess(image.convert("RGB")))

image_batch = torch.stack(processed_images)

query = input("Describe the image you want:")
text_tokens = tokenizer([query])

with torch.inference_mode():
    image_embeddings = model.encode_image(image_batch)
    text_embedding = model.encode_text(text_tokens)
    image_embeddings = image_embeddings / image_embeddings.norm(
        dim = -1,
        keepdim = True
    )
    text_embedding = text_embedding / text_embedding.norm(
        dim = -1,
        keepdim = True
    )

similarity_scores = image_embeddings @ text_embedding.T
similarity_scores = similarity_scores.squeeze(1)

scores, indexes = torch.sort(similarity_scores, descending = True)

for score, index in zip(scores, indexes):
    print(f"{image_path[index].name} : {score.item():.3f}")