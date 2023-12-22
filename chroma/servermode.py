import chromadb

client = chromadb.PersistentClient(path="C:/Users/mauro.cespedes/Desktop/Chroma/start")

chroma_client = chromadb.HttpClient(host='localhost', port=8000)

#-----------Create a collection
# Collections are where you'll store your embeddings, documents, and any additional metadata. 
# You can create a collection with a name:

collection = chroma_client.create_collection(name="my_collection")

#--------- Add some text documents to the collection
# Chroma will store your text, and handle tokenization, embedding, and indexing automatically.

collection.add(
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)

#-------- Query the collection
#You can query the collection with a list of query texts, and Chroma will return the n most similar results. 
#It's that easy!

results = collection.query(
    query_texts=["This is a query document"],
    n_results=2
)

# By default data stored in Chroma is ephemeral making it easy to prototype scripts. 
#It's easy to make Chroma persistent so you can reuse every collection you create and add more documents 
#to it later. It will load your data automatically when you start the client, and save it automatically 
#when you close it. Check out the Usage Guide for more info.

print(type(results))
print("-------------------------------")
print(results)
print("-------------------------------")
print(type(collection))
print("-------------------------------")
print(collection)