import os
import time
from tqdm import tqdm
import gc

from qdrant_client import QdrantClient, models

from .database import VectorDatabase
from ..utils.constants import DATABASES_PATH
from ..utils.utils import load_json_file
from ..utils.embedding_functions import *
from ..utils.logger import init_logger

logger = init_logger(__name__)


class QdrantDatabase(VectorDatabase):

    def __init__(self, config):
        super().__init__(config)
        self.client = QdrantClient(path=os.path.join(DATABASES_PATH, "qdrant", self.embedding_name))


    def embed_problems(self, file):  
        """
            Embed the problems and save them in the database

            Args:
                client: Qdrant client to connect to the database
                file: path to the problems file
                config: configuration dictionary
            
            Returns:
                end_time: time to embed the problems
        """
        # Load the problems to embed
        problems = load_json_file(file)
        basename_file = os.path.basename(file)  # Problems filename will be the key for filtering problems in the database

        # Create the collection only if it doesn't exists
        if not self.client.collection_exists(basename_file) or self.client.count(basename_file).count == 0:
            # Get the embedding function of the embedding model defined
            embd = self.get_emb_function()
            self.client.create_collection(basename_file, optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
                                    quantization_config=models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)),
                                    **embd.params)
        
            # Add problems to the database
            logger.info(f"Adding examples of {basename_file} in the database...")
            start_time = time.time()
            self.add_to_database(embd, problems, basename_file)
            end_time = time.time() - start_time
            del problems
            del embd
            gc.collect()
            return end_time
        else:
            # If problems saved, don't do anythong
            logger.info("All the problems already saved in the database.")
            del problems
            return
    


    def add_to_database(self, embd, problems, collection_name):
        """
            Add the problems to the database

            Args:
                client: Qdrant client to connect to the database
                embd: embedding function
                problems: dictionary of problems
                collection_name: name of the collection to insert the problems
        """

        # List que questions to embed, ids and set metadada
        questions_texts = [problem["question"] for problem in problems.values()]
        question_ids = [key for key in problems.keys()]
        metadata = [{"id": key} for key in question_ids] # Store id as metadata
        
        # Get embeddings of the questions to embed
        embeddings = embd.get_vectors(questions_texts)
        del questions_texts

        if len(embeddings) > 0 and isinstance(embeddings[0], dict):
            # Upload the sparse vectors to the collection
            sparse_vectors = [
                models.PointStruct(
                                id=question_ids[i],
                                payload=metadata[i],
                                vector={
                                    "text": models.SparseVector(indices=list(embeddings[i].keys()), values=list(embeddings[i].values()))
                                }
                ) for i in range(len(question_ids))
            ]

            batch_size = 256
            for i in tqdm(range(0, len(question_ids), batch_size), desc="Adding embeddings to qdrant..."):
                batch_emb = sparse_vectors[i:i+batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch_emb,
                )
        else:
            # Upload the dense vectors to the collection
            batch_size = 256
            for i in tqdm(range(0, len(question_ids), batch_size), desc="Adding embeddings to qdrant..."):
                batch_emb = embeddings[i:i+batch_size]
                batch_questions = question_ids[i:i+batch_size]
                batch_meta = metadata[i:i+batch_size]
                self.client.upload_collection(collection_name, ids=batch_questions, vectors=batch_emb, payload=batch_meta, parallel=8)
        
        logger.info(f"Inserted {len(embeddings)} embeddings to the {collection_name} database.")
        del embeddings, metadata, question_ids
        result = self.client.count(collection_name)
        logger.info(f"Number of total examples in the {collection_name} database after insertions: {result.count}")



    def select_examples(self, problems, solutions, test_collection, database_path):
        """
            From each problem, select the most relevant examples from the existing solutions

            Args:
                problems: dictionary of problems
                solutions: dictionary of solutions
                test_collection: name of the test collection
                database_path: path to the database of examples
                config: configuration dictionary
            Returns:
                outputs: dictionary of the most relevant examples for each problem
        """
        database_name = os.path.basename(database_path)

        # Retrieve test questions embeddings from database
        logger.info("Retrieving embeddings of the test questions...")
        problems_ids = [key for key in problems.keys()]
        retrieved = self.client.retrieve(
            collection_name=test_collection,
            ids=problems_ids,
            with_payload=False,
            with_vectors=True,
        )
 
        # Get ids and embeddings of the result
        retrieved_ids = [r.id for r in retrieved]
        retrieved_vectors = [r.vector for r in retrieved]
        
        # If sparse vectors, convert to the expected format using NamedSparseVector
        if isinstance(retrieved_vectors[0], dict):
            retrieved_vectors = [models.NamedSparseVector(name="text", vector=v["text"]) for v in retrieved_vectors]
        del retrieved
        
        # Set the number of questions to return
        reranker = True if "reranker" in self.config["config"] and self.config["config"]["reranker"] else False
        n_rank = self.config["config"]["reranker"]["n_rank"] if reranker else 1

        # Load solutions to filter only questions that have valid solution
        ids_solutions = list(solutions.keys())

        logger.info("Searching the most relevant examples for each question...")
        # Define the query filter. Filter by filename of the database (or generation file), and valid solutions
        filter_ = models.Filter(
            must=[
                models.FieldCondition(
                    key="id",
                    match=models.MatchAny(
                        any=ids_solutions
                    )
                )
            ]
        )

        # Batch searching
        result = []
        for i in tqdm(range(0, len(retrieved_vectors), 256), desc="Searching examples..."):
            batch_vectors = retrieved_vectors[i:i+256]
            search_queries = [
                models.SearchRequest(vector=emb, 
                                    limit=self.config["config"]["k"]*n_rank,
                                    filter=filter_,
                                    with_payload=True) 
                for emb in batch_vectors
            ]
            r = self.client.search_batch(collection_name=database_name, requests=search_queries)
            result.extend(r)


        # Format the output
        outputs = {}
        for i, value in enumerate(result):
            ids = []
            scores = []
            examples = []
            for item in value:
                ids.append(item.id)
                scores.append(item.score)
                examples.append(item.payload)

            outputs[retrieved_ids[i]] = {
                "ids": ids[::-1], 
                "scores": scores[::-1], 
                "examples": examples[::-1]
            }

        return outputs
