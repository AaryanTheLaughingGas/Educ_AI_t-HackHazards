from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="./llamaparser-example")
documents = reader.load_data()

print(f"Loaded {len(documents)} docs")
print(documents[0].text[:500])

llama_parse_documents = documents