import numpy as np

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    a_length = np.linalg.norm(vector_a)
    b_length = np.linalg.norm(vector_b)

    return dot_product / (a_length * b_length)

query_embedding = np.array([1.0, 0.0, 0.6])

image_embeddings = {
    "dog in a park" : np.array([0.9, 0.0, 0.6]),
    "sports car" : np.array([0.0, 1.0, 0.1]),
    "forest trail" : np.array([0.1, 0.0, 1.0]),
    "cat indoors" : np.array([1.0, 0.0, 0.1]),
}

results = []

for image_name, image_embedding in image_embeddings.items():
    score = cosine_similarity(image_embedding, query_embedding)
    results.append((image_name, score))

results.sort(key = lambda item: item[1], reverse = True)

for image_name, score in results:
    print(f"{image_name}: {score:.3f}")