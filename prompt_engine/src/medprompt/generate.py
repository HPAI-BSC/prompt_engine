import os
import time
from datetime import datetime

from src.utils.constants import *
from src.utils.utils import save_json_file

from .execute import execute
from src.embeddings.qdrant_database import QdrantDatabase
from src.embeddings.chromadb_database import ChromaDatabase

from src.utils.logger import init_logger

logger = init_logger(__name__)


def generate(model, subject, config):
    """
        Generate the CoT examples for the given subject and configuration

        Args:
            model: VLLM model
            subject: subject to generate the examples
            config: configuration dictionary

        Returns:
            out_test_path: path to the generated examples
    """

    # Create a structure to save generation times of each step
    generations_times = {} 
    generate_start_time = time.time()

    # Load main parameters, configuration and sampling_params
    configuration = config["config"]
    working_dir = configuration["working_dir"] if "working_dir" in configuration else None
    logger.info(f"Working directory: {working_dir}")

    if "sampling_params" in config:
        sampling_params = config["sampling_params"]
    else:
        sampling_params = None
    
    model_name = model.model_name

    # Name and path of the input dataset
    input_name = f"{configuration['dataset']}_{subject}" if subject is not None and subject != "" else configuration["dataset"]
    input_path = os.path.join(DATASETS_PATH, configuration["dataset"], input_name)

    # Used to store the output in a structured format (add a new level of subfolder if subject exits)
    dataset_path = configuration["dataset"]
    if subject is not None:
        dataset_path = os.path.join(dataset_path, subject)

    # Set default parameters for medprompt
    if "ensembles" not in configuration:
        configuration["ensembles"] = 5
    if "k" not in configuration:
        configuration["k"] = 5
    if "shuffle" not in configuration:
        configuration["shuffle"] = True
    if "database" not in configuration:
        configuration["database"] = None
    if "reranker" not in configuration:
        configuration["reranker"] = None
    overwrite = configuration["overwrite"] if "overwrite" in configuration else True

    if "embedding" in configuration:
        # Medprompt execution
        embedding_name = os.path.basename(configuration["embedding"])
        if embedding_name in ["random", "static"]:
            datastore = None
        else:
            logger.info(f"Loading client database: {embedding_name}...")
            if "vector_database" in configuration and configuration["vector_database"] == "qdrant":
                datastore = QdrantDatabase(config)
            else:
                datastore = ChromaDatabase(config)  # If not specified, use ChromaDB as default
            logger.info("Database loaded")
    
        # Check if database should be used
        if configuration["database"] is None:
            # Save the embeddings of the validation problems in the ChromaDB database
            logger.info("Embedding validation problems...")
            embd_val_time = datastore.embed_problems(input_path + "_val.json") if datastore else None
            if embd_val_time:
                generations_times["embd_val_problems"] = embd_val_time
            
            # No database provided. Generate validation examples using the LLM
            out_val_path = os.path.join(working_dir, "outputs", model_name, "val", dataset_path, str(configuration["ensembles"]), f"{configuration['k']}k")

            if overwrite or not os.path.exists(os.path.join(out_val_path, "generations.json")):
                logger.info(f"GENERATING VALIDATION CoT EXAMPLES...\nDataset: {input_name}_val.json\nExamples will be saved here: {out_val_path}\n")
                val_start_time = test_start_time = time.time()
                val_times = execute(datastore, model, dataset_path=input_path + "_val.json", output_path=out_val_path, examples=None, database_filename=None, configuration=configuration, sampling_params=sampling_params)
                
                # Execute returns execution times of each step
                generations_times["validation_gen_time"] = {
                    "total_time": time.strftime('%H:%M:%S', time.gmtime(time.time() - val_start_time)),
                    "stages": val_times
                }
            
            examples = out_val_path # Set the output of the validation examples, as input test examples
            chromda_db_filename = os.path.basename(input_path) + "_val.json"
        else:
            examples = os.path.join(working_dir, "databases", configuration["database"])
            if not examples.endswith(".json"):
                examples += ".json"
                
            chromda_db_filename = os.path.basename(examples)

            # Save the embeddings of the database problems in the ChromaDB database
            embd_database_time = datastore.embed_problems(examples) if datastore else None
            if embd_database_time:
                generations_times["embd_DB_problems"] = embd_database_time
            logger.info(f"USING DATABASE AS EXAMPLES: {examples}\n")

        # Define the name of the last output level. If database and/or reanker used, append to the name
        k_out = f"{configuration['k']}k"
        if configuration["database"] is not None:
            k_out += f"_{configuration['database']}"
        if configuration["reranker"] is not None:
            reranker_name = os.path.basename(configuration["reranker"]["path"])
            k_out += f"_{reranker_name}"
        
        # Set the output path of the Medprompt generations
        out_test_path = os.path.join(working_dir, "outputs", model_name, embedding_name, dataset_path, str(configuration["ensembles"]), k_out)

        # Save the embeddings of the test problems in the database
        logger.info(f"Embedding test problems: {input_path}_test.json")
        embd_test_time = datastore.embed_problems(input_path + "_test.json") if datastore else None
        if embd_test_time:
            generations_times["embd_test_problems"] = embd_test_time

    else:
        # SC-COT execution
        # If SC-COT, database, embedding and reranker are not needed
        configuration["embedding"] = None
        configuration["database"] = None
        configuration["reranker"] = None

        # Set the output path of the SC-COT generations
        out_test_path = os.path.join(working_dir, "outputs", model_name, "SC-COT", dataset_path, str(configuration["ensembles"]), f"{configuration['k']}k")

        # If SC-COT, no examples are needed
        examples = None
        chromda_db_filename = None
        datastore = None
        

    logger.info(f"Generating CoT of test examples...\nDataset: {input_name}_test.json\nExamples will be saved here: {out_test_path}\n")
    test_start_time = time.time()

    # Execute test generations
    test_times = execute(datastore, model, dataset_path=input_path + "_test.json", output_path=out_test_path, examples=examples, database_filename=chromda_db_filename, configuration=configuration, sampling_params=sampling_params)

    if datastore:
        del datastore

    generations_times["test_gen_time"] = {
        "total_time": time.strftime('%H:%M:%S', time.gmtime(time.time() - test_start_time)),
        "stages": test_times
    }
    generations_times["total_gen_time"] = time.strftime('%H:%M:%S', time.gmtime(time.time() - generate_start_time))
    times_filename = f"times_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"

    # Save a file with the executions time information in the same folder as the output
    save_json_file(os.path.join(out_test_path, times_filename), generations_times)
    return out_test_path
