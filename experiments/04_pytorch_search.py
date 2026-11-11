import torch
import torch.nn.functional as F

image_names = [
    "dog in a park",
    "sports car",
    "forest trail",
    "cat indoors",
]

image_embeddings = torch.tensor([
    [0.9, 0.0, 0.7],
    [0.0, 1.0, 0.1],
    [0.1, 0.0, 1.0],
    [1.0, 0.0, 0.1],
])

query_embedding = torch.tensor([1.0, 0.0, 0.6])

image_embeddings = F.normalize(image_embeddings, dim = 1)
query_embedding = F.normalize(query_embedding, dim = 0)

similarity_scores = image_embeddings @ query_embedding

scores, indexes = torch.sort(similarity_scores, descending= True)

for score, index in zip(scores, indexes):
    print(f"{image_names[index]}: {score.item():.3f}")