from pathlib import Path
from visual_search import ClipEmbedder, ExactImageIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE_PATH = (
    PROJECT_ROOT / "data" / "index" / "image_index.pt"
)

embedding_model = ClipEmbedder()

exact_index = ExactImageIndex(
    index_file_path = INDEX_FILE_PATH,
    project_root_path= PROJECT_ROOT
)

if (
    exact_index.model_name != embedding_model.model_name or
    exact_index.pretrained != embedding_model.pretrained
):
    raise ValueError(
        "The index was created with a different CLIP model."
        "Run build_index.py again."
    )

query_text = input("Describe the image you want: ")
query_embedding = embedding_model.encode_text(query_text)

search_results = exact_index.search(
    query_embedding = query_embedding,
    top_k = 5
)

print(f"Results for: {query_text}\n")

for image_path, similarity_score in search_results:
    print(f"{image_path.name} : {similarity_score:.3f}")