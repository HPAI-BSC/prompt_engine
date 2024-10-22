import os

from src.utils.constants import *
from src.utils.utils import *
from src.embeddings.qdrant_database import QdrantDatabase
from src.embeddings.chromadb_database import ChromaDatabase

from src.qa.execute import execute

from src.utils.logger import init_logger

logger = init_logger(__name__)

def generate(model, subject, config):
    configuration = config["config"]

    # Set default parameters
    if "sampling_params" in config:
        sampling_params = config["sampling_params"]
    else:
        sampling_params = None
    
    if "ensembles" not in configuration:
        configuration["ensembles"] = 5
    if "k" not in configuration:
        configuration["k"] = 5
    if "overwrite" in configuration:
        overwrite = configuration["overwrite"]
    else:
        overwrite = True

    if "embedding" not in configuration or configuration["embedding"] is None:
        raise ValueError("Embedding not specified in configuration")
    
    embedding_name = os.path.basename(configuration["embedding"])
    model_name = os.path.basename(config["vllm"]["model"])


    # Load the database client
    logger.info(f"Loading client database: {embedding_name}...")
    if "vector_database" in configuration and configuration["vector_database"] == "qdrant":
        datastore = QdrantDatabase(config)
    else:
        datastore = ChromaDatabase(config)  # If not specified, use ChromaDB as default
    logger.info("Database client loaded")


    # Load and embed the database
    if "database" not in configuration or configuration["database"] is None:
        database_examples = None
    else:
        database_path = os.path.join(DATABASES_PATH, configuration["database"])
        if not database_path.endswith(".json"):
            database_path += ".json"
        datastore.embed_problems(database_path)
        database_examples = (os.path.basename(database_path), load_qa_database(database_path))


    # Load and embed the input dataset
    input_name = f"{configuration['dataset']}_{subject}" if subject is not None and subject != "" else configuration["dataset"]
    input_path = os.path.join(DATASETS_PATH, configuration["dataset"], input_name + "_test.json")
    datastore.embed_problems(input_path)
    test_dataset = (input_name + "_test.json", load_qa_file(input_path))

    # Set the output path
    k_out = f"{configuration['k']}k"
    if "database" in configuration and configuration["database"] is not None:
        k_out += f"_{configuration['database']}"
    if "reranker" in configuration and configuration["reranker"] is not None:
        reranker_name = os.path.basename(configuration["reranker"]["path"])
        k_out += f"_{reranker_name}"

    out_test_path = os.path.join(GENERATIONS_PATH, model_name, "QA", configuration["final_answer"], embedding_name, input_name, str(configuration["ensembles"]), k_out)
    os.makedirs(out_test_path, exist_ok=True)

    # Set default sampling parameters
    if "max_tokens" not in sampling_params:
        sampling_params["max_tokens"] = 1000
    
    if "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.01
    if sampling_params["temperature"] > 1:
        sampling_params["temperature"] = 1
        
    execute(datastore, model, test_dataset, database_examples, out_test_path, configuration, sampling_params)

    return out_test_path




    
