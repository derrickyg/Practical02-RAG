import redis
import chromadb
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections
from pymilvus import utility

class RedisManager:
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        self.db = self.connect_to_redis(host, port, db, password)

    def connect_to_redis(self, host, port, db, password):
        try:
            # Create a Redis connection
            redis_conn = redis.Redis(host=host, port=port, db=db, password=password)
            # Check connection
            redis_conn.ping()
            print("Connected to Redis successfully!")
            return redis_conn
        except redis.ConnectionError as e:
            print(f"Failed to connect to Redis: {e}")
            return None

    def set_key(self, key, value):
        if self.db is not None:
            try:
                self.db.set(key, value)
                print(f"Key {key} set successfully.")
            except Exception as e:
                print(f"Failed to set key {key}: {e}")

    def get_key(self, key):
        if self.db is not None:
            try:
                value = self.db.get(key)
                print(f"Value for key {key}: {value}")
                return value
            except Exception as e:
                print(f"Failed to retrieve key {key}: {e}")
                return None


class ChromaManager:
    def __init__(self, host='localhost', port=1234, api_key=None):
        self.db = self.connect_to_chroma(host, port, api_key)

    def connect_to_chroma(self, host, port, api_key):
        try:
            # Assuming 'ChromaClient' is a part of the Chroma client library
            from chroma_client import ChromaClient
            chroma_conn = ChromaClient(host=host, port=port, api_key=api_key)
            # Example of a hypothetical ping method to check connection
            if chroma_conn.ping():
                print("Connected to Chroma successfully!")
            return chroma_conn
        except Exception as e:
            print(f"Failed to connect to Chroma: {e}")
            return None

    def add_vector(self, key, vector):
        if self.db is not None:
            try:
                self.db.insert_vector(key, vector)
                print(f"Vector with key {key} added successfully.")
            except Exception as e:
                print(f"Failed to add vector {key}: {e}")

    def search_vector(self, vector, top_k=10):
        if self.db is not None:
            try:
                results = self.db.search_vector(vector, top_k=top_k)
                print(f"Search results: {results}")
                return results
            except Exception as e:
                print(f"Failed to search for vector: {e}")
                return None
            

class MilvusManager:
    def __init__(self, host='localhost', port='19530', collection_name='example_collection'):
        self.collection_name = collection_name
        self.connect_to_milvus(host, port)
        self.setup_collection()

    def connect_to_milvus(self, host, port):
        # Connect to Milvus
        connections.connect(alias="default", host=host, port=port)
        print(f"Connected to Milvus on {host}:{port}")

    def setup_collection(self):
        # Define fields for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
        ]
        schema = CollectionSchema(fields, description="Test collection")
        if not utility.has_collection(self.collection_name):
            # Create collection if it doesn't exist
            collection = Collection(name=self.collection_name, schema=schema)
            print(f"Collection {self.collection_name} created.")
        else:
            print(f"Collection {self.collection_name} already exists.")

    def insert_vectors(self, vectors):
        # Insert vectors into the collection
        collection = Collection(self.collection_name)
        ids = collection.insert([vectors])
        print(f"Inserted {len(ids)} vectors.")
        # Flush data to ensure consistency
        collection.load()

    def search_vectors(self, query_vectors, top_k=5):
        # Search for similar vectors
        collection = Collection(self.collection_name)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(query_vectors, "embedding", search_params, limit=top_k, expr=None)
        print("Search results:")
        for result in results[0]:
            print(f"ID: {result.id}, Distance: {result.distance}")
        return results

def main():
    # Initialize Redis Manager
    redis_manager = RedisManager()
    # Example usage
    redis_manager.set_key('test_key', 'Hello Redis!')
    value = redis_manager.get_key('test_key')



    # Initialize Chroma Manager
    chroma_manager = ChromaManager()
    # Example usage
    test_vector = [0.1, 0.2, 0.3, 0.4]
    chroma_manager.add_vector('test_vector', test_vector)
    results = chroma_manager.search_vector(test_vector)



    # Example usage of MilvusManager
    milvus_manager = MilvusManager()
    # Example vector
    vectors = [[0.1, 0.2, 0.3, 0.4] * 32]  # 128 dimensions
    milvus_manager.insert_vectors(vectors)
    # Search for similar vectors
    search_results = milvus_manager.search_vectors(vectors)

if __name__ == "__main__":
    main()
