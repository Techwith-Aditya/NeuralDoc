import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

# Model configuration
embedder = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

def md_to_json(md_file_path):
    """Convert markdown file to hierarchical JSON structure with title/content separation"""
    sections = []
    current_section = {"main title": "", "section title": "", "content": []}
    main_title = ""
    
    with open(md_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith("# "):
            if current_section["section title"]:
                sections.append(current_section)
            main_title = line[2:].strip()
            current_section = {"main title": main_title, "section title": "", "content": []}
        elif line.startswith("## "):
            if current_section["section title"]:
                sections.append(current_section)
            current_section = {"main title": main_title, "section title": line[3:].strip(), "content": []}
        else:
            current_section["content"].append(line)
    
    if current_section["section title"]:
        sections.append(current_section)
    
    for section in sections:
        section["content"] = "\n".join(section["content"]).strip()
    
    return sections

def create_or_load_collection():
    """Manage Milvus collection lifecycle with schema versioning"""
    collection_name = "md_embeddings"
    connections.connect("default", host="localhost", port="19530")

    if utility.has_collection(collection_name):
        return Collection(name=collection_name)
    
    # Schema design
    schema = CollectionSchema([
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="main_title_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="section_title_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="content_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    ], description="Embeddings collection for markdown files")

    return Collection(name=collection_name, schema=schema)

def create_indexes(collection):
    """Create HNSW indexes for efficient vector search"""
    index_params = {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index("main_title_embedding", index_params)
    collection.create_index("section_title_embedding", index_params)
    collection.create_index("content_embedding", index_params)

def generate_embeddings(text):
    """Generate 1024-dim embeddings with fallback for empty text"""
    return embedder.encode(text).tolist() if text else [0.0] * 1024

def insert_data_into_milvus(collection, json_data):
    """Batch insert with embeddings for titles, subtitles, and content"""
    main_title_embeddings = []
    section_title_embeddings = []
    content_embeddings = []
    texts = []
    
    for data in json_data:
        main_title_embeddings.append(generate_embeddings(data["main title"]))
        section_title_embeddings.append(generate_embeddings(data["section title"]))
        content_embeddings.append(generate_embeddings(data["content"]))
        texts.append(data["content"])
    
    collection.insert([main_title_embeddings, section_title_embeddings, content_embeddings, texts])
    collection.flush()

def query(query_text, anns_field="section_title_embedding", limit=1, threshold=0.90):
    """Semantic search with similarity threshold filtering"""
    connections.connect("default", host="localhost", port="19530")
    collection = Collection(name="md_embeddings")
    collection.load()

    results = collection.search(
        data=[generate_embeddings(query_text)],
        anns_field=anns_field,
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=limit,
        output_fields=["text"]
    )

    return [{"text": hit.entity.get("text"), "similarity": hit.distance}
            for res in results for hit in res if hit.distance >= threshold]

def main():
    # Pipeline execution flow
    json_data = md_to_json("md_output/output1.md")
    
    with open("cnn1.json", "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4)
    
    collection = create_or_load_collection()
    create_indexes(collection)  # Indexes created after collection creation
    insert_data_into_milvus(collection, json_data)
    print(query("Introduction"))

if __name__ == "__main__":
    main()
