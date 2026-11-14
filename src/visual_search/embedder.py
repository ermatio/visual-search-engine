from pathlib import Path
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

class ClipEmbedder:
    """Generate normalized CLIP embeddings for text and images"""

    def __init__(
            self,
            model_name: str = "ViT-B-32-quickgelu",
            pretrained: str = "openai",
            device_type: str = "cpu"
    ) -> None:

        self.device = torch.device(device_type)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained = pretrained
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model = self.model.to(self.device)

        self.model.eval()

    def encode_text(self, text: str) -> Tensor:
        """Convert one text query into a normalized embedding."""
        tokens = self.tokenizer([text]).to(self.device)

        with torch.inference_mode():
            text_embedding = self.model.encode_text(tokens)

        return F.normalize(text_embedding, dim = -1)

    def encode_images(self, image_paths: list[Path]) -> Tensor:
        """Convert one or more image files into normalized embeddings."""
        if not image_paths:
            raise ValueError("images not found")

        processed_images = []

        for image_path in image_paths:
            with Image.open(image_path) as image:
                processed_images.append(self.preprocess(image.convert("RGB")))

        image_batch = torch.stack(processed_images).to(self.device)

        with torch.inference_mode():
            image_embeddings = self.model.encode_image(image_batch)

        return F.normalize(image_embeddings)

