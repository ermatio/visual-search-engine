import numpy as np

image_embedding = np.array([0.8, 0.1, 0.6])

print("Embedding:", image_embedding)
print("Shape:", image_embedding.shape)
print("Dimensions:",image_embedding.ndim)
query_embedding = np.array([1.0, 0.0, 0.5])

dot_product = np.dot(query_embedding, image_embedding)
query_length = np.linalg.norm(query_embedding)
image_length = np.linalg.norm(image_embedding)

similarity = dot_product / (query_length * image_length)

print("dot product:", dot_product)
print("length of query:", query_length)
print("length of image:", image_length)
print("Cosine similarity:", similarity)