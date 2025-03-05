import os
import time
from tqdm import tqdm
import gc

from chromadb import PersistentClient

from .database import VectorDatabase
from src.utils.constants import DATABASES_PATH
from src.utils.utils import load_json_file
from src.utils.embedding_functions import *
from src.utils.logger import init_logger

logger = init_logger(__name__)


class ChromaDatabase(VectorDatabase):
    def __init__(self, config):
        super().__init__(config)
        save_dir = config["config"]["working_dir"] if "working_dir" in config["config"] else DATABASES_PATH
        self.client = PersistentClient(path=os.path.join(save_dir, "vector_databases", "chromadb", self.embedding_name))



    def embed_problems(self, file):
        """
            Embed the problems and save them in the database

            Args:
                client: chromadb client to connect to the database
                file: path to the problems file
                config: configuration dictionary
            
            Returns:
                end_time: time to embed the problems
        """
        
        # Load the problems to embed
        problems = load_json_file(file)
        basename_file = os.path.basename(file)  # Problems filename will be the key for filtering problems in the database
        
        # Read the collection of the corresponding embedding
        collection = self.client.get_or_create_collection(name=basename_file, 
                                                    metadata={"hnsw:space": "cosine"})
        
        # Load the problems
        problems = load_json_file(file)
        basename_file = os.path.basename(file)  # Problems filename will be the key for filtering problems in the database

        if collection.count() < len(problems):
            # Add problems to the database
            logger.info(f"Adding examples of {basename_file} in the database...")
            start_time = time.time()
            self.add_to_database(problems, collection, basename_file)
            end_time = time.time() - start_time
            del problems
            gc.collect()
            torch.cuda.empty_cache()
            return end_time
        else:
            logger.info("All the problems already saved in the database.")
            del problems
            return


    def add_to_database(self, problems, collection, file):
        """
            Add the problems to the database

            Args:
                problems: dictionary of problems
                collection: chromadb collection
                file: name of the file. Will be the key for grouping them
        """
        
        questions_texts = [problem["question"] for problem in problems.values()]
        questions_ids = list(problems.keys())
        metadata = [{"id": key} for key in problems.keys()]
        
        # Embed the questions
        embd = self.get_emb_function()
        # Add the embeddings to the database
        logger.info(f"Adding {len(questions_ids)} examples to the database...")
        logger.info(f"Number of examples in the database before: {collection.count()}")
        # upsert() will add documents if don't exits or update if they exits.
        batch_size = 256
        for i in tqdm(range(0, len(questions_ids), batch_size), desc="Adding embeddings to ChromaDB..."):
            batch_questions_ids = questions_ids[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            questions_to_embed = questions_texts[i:i+batch_size]
            batch_emb = embd.get_vectors(questions_to_embed)
            collection.upsert(embeddings=batch_emb, ids=batch_questions_ids, metadatas=batch_metadata)
            del batch_emb, batch_metadata, batch_questions_ids

        del embd, metadata, questions_ids
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"Number of examples in the database after: {collection.count()}")


    def select_examples(self, problems, solutions, problems_collection, database_path):
        """
            From each problem, select the most relevant examples from the existing solutions

            Args:
                client: chromadb client to connect to the database
                problems: dictionary of problems
                solutions: dictionary of solutions
                dataset_path: path to the dataset
                config: configuration dictionary

            Returns:
                outputs: dictionary of the most relevant examples for each problem
        """
        

        # Retrieve test questions embeddings from database
        problems_collection = self.client.get_collection(name=problems_collection)
        logger.info("Retrieving embeddings of the test questions...")
        problems_ids = [i for i in problems.keys()]
        embeddings = problems_collection.get(ids=problems_ids, include=["embeddings"])["embeddings"]

        # Set the number of questions to return
        reranker = True if "reranker" in self.config["config"] and self.config["config"]["reranker"] else False
        n_rank = self.config["config"]["reranker"]["n_rank"] if reranker else 1

        # Filter only examples that have at least a correct solution
        ids_solutions = list(solutions.keys())  
        
        logger.info("Searching the most relevant examples for each question...")
        # Query K examples of the dataset_name
        database_name = os.path.basename(database_path)
        database_collection = self.client.get_collection(name=database_name)

        # If all the examples are needed, set where to None. This will speed up the query
        if len(ids_solutions) == database_collection.count():
            where = None
        else:
            where = {"id": {"$in": ids_solutions}}
            
        result = database_collection.query(query_embeddings=embeddings,
                                  n_results=self.config["config"]["k"]*n_rank, 
                                  where=where, 
                                  include=["distances", "metadatas"])
        
        # Get the most relevant examples for each problem
        outputs = {}
        for i in range(len(result["ids"])):
            outputs[problems_ids[i]] = {
                "ids": result["ids"][i][::-1], 
                "scores": [1-r for r in result["distances"][i][::-1]], # Convert distance to similarity
                "examples": result["metadatas"][i][::-1]}

        return outputs
