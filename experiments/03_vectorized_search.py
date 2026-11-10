import numpy as np

image_names = [
    "dog in a park",
    "sports car",
    "forest trail",
    "cat indoors",
]

image_embeddings = np.array([
    [0.9, 0.0, 0.7],
    [0.0, 1.0, 0.1],
    [0.1, 0.0, 1.0],
    [1.0, 0.0, 0.1],    
])

query_embedding = np.array([1.0, 0.0, 0.6])

query_embedding = query_embedding / np.linalg.norm(query_embedding)

image_length = np.linalg.norm(
    image_embeddings,
    axis=1,
    keepdims = True,
)

image_embeddings = image_embeddings / image_length

similarity_scores = image_embeddings @ query_embedding
ranked_indexes = np.argsort(similarity_scores)[::-1]

for index in ranked_indexes:
    print(f"{image_names[index]}: {similarity_scores[index]:.3f}")
    