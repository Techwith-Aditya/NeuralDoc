import json
import os
import sys

from dotenv import load_dotenv
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, list_collections
from sentence_transformers import SentenceTransformer  # Embedding model

class MilvusEmbeddingManager:
    def __init__(self, host="host.docker.internal", port="19530"):
        self.host = host
        self.port = port
        load_dotenv()
        self.embedder = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')
        connections.connect("default", host=host, port=port)
        print("Connected to Milvus.")

    def create_or_load_collection(self, collection_name):
        if collection_name in list_collections():
            print(f"Loading existing collection: {collection_name}")
            return Collection(name=collection_name)
        
        # Define schema for new collection
        schema = CollectionSchema([
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="main_title_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="section_title_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="sub_heading_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="content_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="sub_heading", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1024)
        ], description=f"Embeddings for {collection_name}")

        print(f"Creating new collection: {collection_name}")
        return Collection(name=collection_name, schema=schema)

    def generate_embeddings(self, text):
        """Generate text embeddings"""
        return self.embedder.encode(text) if text else [0.0] * 1024

    def process_and_insert_json(self, json_file):
        """Process JSON data and insert into Milvus"""
        collection_name = os.path.splitext(os.path.basename(json_file))[0]
        collection = self.create_or_load_collection(collection_name)
        global_id = 1
        record_count = 0

        with open(json_file, "r", encoding="utf-8") as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                return

        def process_node(node):
            nonlocal global_id, record_count
            node_id = global_id
            global_id += 1

            metadata = node.get("metadata", {})
            main_title = metadata.get("main title", "")
            section_title = metadata.get("section title", "")
            sub_heading = metadata.get("sub heading", "").strip()
            image_path = metadata.get("image", "No image available")
            content = node["metadata"]["caption"] if "image" in metadata else node.get("content", "")

            # Generate embeddings
            embeddings = [
                self.generate_embeddings(main_title),
                self.generate_embeddings(section_title),
                self.generate_embeddings(sub_heading),
                self.generate_embeddings(content)
            ]

            collection.insert([[
                node_id,
                *embeddings,
                content,
                sub_heading,
                image_path
            ]])
            record_count += 1

            for sub_node in node.get("subheadings", []):
                process_node(sub_node)

        for node in json_data:
            process_node(node)

        print(f"Inserted {record_count} records into {collection_name}")

    def create_indexes(self, collection_name):
        """Create HNSW indexes for vector fields"""
        collection = self.create_or_load_collection(collection_name)
        collection.flush()
        index_params = {"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}}

        for field in ["main_title_embedding", "section_title_embedding", 
                     "sub_heading_embedding", "content_embedding"]:
            collection.create_index(field, index_params)
        
        print(f"Indexes created for {collection_name}")

    def query(self, query_text, anns_field="sub_heading_embedding", limit=5, threshold=0.85):
        """Search across collections with similarity threshold"""
        combined_results = {}
        print(f"Search field: {anns_field}")

        for collection_name in list_collections():
            collection = self.create_or_load_collection(collection_name)
            collection.load()
            query_embedding = self.generate_embeddings(query_text)
            search_params = {"metric_type": "IP", "params": {"ef": 128}}

            is_content_search = anns_field == "content_embedding"
            output_fields = ["text", "image_path"] if is_content_search else ["text", "sub_heading"]

            results = collection.search(
                data=[query_embedding],
                anns_field=anns_field,
                param=search_params,
                limit=limit * 2,
                output_fields=output_fields
            )

            filtered = []
            for res in results:
                for hit in res:
                    if hit.distance < threshold:
                        continue
                    
                    result = {
                        "text": hit.entity.get("text"),
                        "similarity": hit.distance,
                        "collection_name": collection_name
                    }
                    
                    if is_content_search:
                        result["image"] = hit.get("image_path") or "No image"
                    else:
                        result["sub_heading"] = hit.entity.get("sub_heading")

                    filtered.append(result)

            combined_results[collection_name] = filtered[:limit]

        return combined_results

    def perform_default_queries(self):
        """Predefined common queries"""
        default_queries = ["Introduction", "Abstract", "Conclusion", 
                          "References", "Methodology", "Results"]
        organized_results = {query: {} for query in default_queries}

        for collection_name in list_collections():
            collection = self.create_or_load_collection(collection_name)
            collection.load()

            for query_text in default_queries:
                results = collection.search(
                    data=[self.generate_embeddings(query_text)],
                    anns_field="sub_heading_embedding",
                    param={"metric_type": "IP", "params": {"ef": 128}},
                    limit=1,
                    output_fields=["text", "sub_heading"]
                )

                for res in results:
                    for hit in res:
                        organized_results[query_text].setdefault(collection_name, []).append({
                            "text": hit.entity.get("text"),
                            "similarity": hit.distance
                        })

        return organized_results

if __name__ == "__main__":
    manager = MilvusEmbeddingManager()
    json_file = 'output/cnn1.json'

    manager.process_and_insert_json(json_file)
    manager.create_indexes(os.path.splitext(os.path.basename(json_file))[0])

    if user_query := input("Enter your search query: ").strip():
        results = manager.query(user_query, anns_field="sub_heading_embedding", limit=5)
        print("Search Results:", json.dumps(results, indent=4))
        
        with open('keywordbased.txt','w',encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        default_results = manager.perform_default_queries()
        with open('default.txt','w',encoding='utf-8') as f:
            json.dump(default_results, f, indent=4)
