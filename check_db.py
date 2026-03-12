import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("langchain")

print(f"Total chunks: {collection.count()}")
print("\n--- Sample chunks ---")
results = collection.peek(3)
for doc in results["documents"]:
    print(doc[:300])
    print("---")