import os
from .embedding_functions import *

class VectorDatabase:

    def __init__(self, config):
        self.config = config
        self.embedding_path = config["config"]["embedding"]
        self.embedding_name = os.path.basename(self.embedding_path)


    def embed_problems(self, file):
        raise NotImplementedError
    
    def select_examples(self, problems, solutions, test_collection, database_path):
        raise NotImplementedError
    
    def get_emb_function(self):
        """
            Get the embedding function based on the embedding model

            Args:
                embedding_path: path to the embedding model

            Returns:    
                embedding_function: embedding function
        """
        if self.embedding_name == "UAE-Large-V1":
            return UaeLargeEmbedding(embedding_path=self.embedding_path)
        elif self.embedding_name in ["BiomedNLP-BiomedBERT-base-uncased-abstract", "MedCPT-Query-Encoder"]:
            return AutoModelEmbedding(embedding_path=self.embedding_path)
        elif "splade" in self.embedding_name:
            return SpladeEmbedding(embedding_path=self.embedding_path)
        else:
            # If the embedding is not one of the above, use SentenceTransformer
            return SentenceTransformersEmbedding(embedding_path=self.embedding_path)