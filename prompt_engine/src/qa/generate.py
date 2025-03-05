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
    working_dir = configuration["working_dir"] if "working_dir" in configuration else None

    # Set default parameters
    if "sampling_params" in config:
        sampling_params = config["sampling_params"]
    else:
        sampling_params = None
    
    if "ensembles" not in configuration:
        configuration["ensembles"] = 5
    if "k" not in configuration:
        configuration["k"] = 5
    if "final_answer" not in configuration:
        configuration["final_answer"] = "merge"

    if "embedding" not in configuration or configuration["embedding"] is None:
        logger.info("Embedding not specified. Setting embedding to 'static'")
        configuration["embedding"] = "static"
    
    embedding_name = os.path.basename(configuration["embedding"])
    model_name = model.model_name

    # Load the database client
    logger.info(f"Loading client database: {embedding_name}...")
    if embedding_name in ["static", "random"]:
        datastore = None
    elif "vector_database" in configuration and configuration["vector_database"] == "qdrant":
        datastore = QdrantDatabase(config)
    else:
        datastore = ChromaDatabase(config)  # If not specified, use ChromaDB as default
    logger.info("Database client loaded")


    # Load and embed the database
    if "database" not in configuration or configuration["database"] is None:
        database_examples = None
    else:
        database_path = working_dir if working_dir is not None else GENERATIONS_PATH
        database_path = os.path.join(database_path, "databases", configuration["database"])
        if not database_path.endswith(".json"):
            database_path += ".json"
        database_examples = (os.path.basename(database_path), load_qa_database(database_path))
        
        if datastore:
            datastore.embed_problems(database_path)


    # Load and embed the input dataset
    input_name = f"{configuration['dataset']}_{subject}" if subject is not None and subject != "" else configuration["dataset"]
    input_path = os.path.join(DATASETS_PATH, configuration["dataset"], input_name + "_test.json")
    test_dataset = (input_name + "_test.json", load_qa_file(input_path))
    if datastore:
        datastore.embed_problems(input_path)

    # Set the output path
    k_out = f"{configuration['k']}k"
    if "database" in configuration and configuration["database"] is not None:
        k_out += f"_{configuration['database']}"
    if "reranker" in configuration and configuration["reranker"] is not None:
        reranker_name = os.path.basename(configuration["reranker"]["path"])
        k_out += f"_{reranker_name}"

    out_test_path = working_dir if working_dir is not None else GENERATIONS_PATH
    out_test_path = os.path.join(out_test_path, "outputs", model_name, "QA", configuration["final_answer"], embedding_name, input_name, str(configuration["ensembles"]), k_out)
    os.makedirs(out_test_path, exist_ok=True)

    # Set default sampling parameters
    if "max_tokens" not in sampling_params:
        sampling_params["max_tokens"] = 1000
        
    execute(datastore, model, test_dataset, database_examples, out_test_path, configuration, sampling_params)

    return out_test_path