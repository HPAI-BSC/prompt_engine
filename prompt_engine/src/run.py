import argparse
import os
import importlib

from src.utils.model import Model
from src.utils.openai_model import OpenAIModel
from src.utils.constants import *
from src.utils.utils import load_config, filter_non_empty

from src.medprompt.generate import generate
from src.medprompt.evaluate import evaluate_from_path

from src.utils.logger import init_logger

logger = init_logger(__name__)

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument(
        "configuration", type=str, help="Path of the YAML configuration file"
    )
    p.add_argument("--ip-server", type=str, default="localhost", help="IP address of an already started vLLM server")
    p.add_argument("--port", type=str, default="6378", help="Port of the vLLM server")
    return p.parse_args()


# Check the format of the configuration file
def check_configuration(config):
    assert "dataset" in config["config"], "ERROR: Dataset not specified in configuration."
    assert config["config"]["dataset"] in VALID_DATASETS, "ERROR: Not a valid dataset: " + config["config"]["dataset"]
    
    if "reranker" in config["config"]:
        assert os.path.basename(config["config"]["reranker"]["path"]) in VALID_RERANKERS


def format_dataset_subjects(conf):
    dataset = conf["dataset"]
    subjects = conf["subject"] if "subject" in conf else None
    if dataset == "mmlu":
        if subjects is None or subjects == "all":
            return VALID_SUBJECTS
        elif "," in subjects:
            return [s for s in subjects.split(",") if s in VALID_SUBJECTS]
        else:
            if subjects not in VALID_SUBJECTS:
                raise ValueError(f"Not a valid mmlu subject: ", subjects)
            return [subjects]
    else:
        return [None]


def prepare():
    args = parse_arguments()    # Configuration file should be defined in the script parameters

    # Check valid YAML file provided
    assert args.configuration.endswith(".yaml"), "Model must be a YAML with the vllm model configuration"
    assert os.path.exists(args.configuration), "Configuration path doens't exists: " + args.configuration

    # Load configuration
    config = load_config(args.configuration)
    assert "vllm" or "openai" in config, "ERROR: 'vllm' or 'openai' parameters not included in configuration. Cannot run without a model."
    assert "config" in config, "ERROR: 'config' parameters not included in configuration."
    if "vllm" in config:
        config["vllm"] = filter_non_empty(config["vllm"])
    if "openai" in config:
        config["openai"] = filter_non_empty(config["openai"])
    config["config"] = filter_non_empty(config["config"])
    config["sampling_params"] = filter_non_empty(config["sampling_params"])
    check_configuration(config)

    subjects = format_dataset_subjects(config["config"])    # If mmlu, get all subjects to execute
    ip_server = f"{args.ip_server}:{args.port}"
    return subjects, config, ip_server

def run(subjects, config, ip="localhost:6378"):
    # Load the model using VLLM and config parameters
    model = None
    logger.info("\nStarting execution with the following configuration:")
    logger.info(config)
    for subject in subjects:
        if model is None:
            if "vllm" in config:
                model = Model(config["vllm"])
            elif "openai" in config:
                model = OpenAIModel(config["openai"], ip)
            else:
                raise ValueError("Model not specified in configuration. Please specify 'vllm' or 'openai' model parameters.")

        ex_type = EXAMPLES_CONVERSION[config["config"]["dataset"]]
        
        if ex_type == "qa":
            module_generate = importlib.import_module("src.qa.generate")
            module_evaluate = importlib.import_module("src.qa.evaluate")
        else:
            module_generate = importlib.import_module("src.medprompt.generate")
            module_evaluate = importlib.import_module("src.medprompt.evaluate")

        output_path = module_generate.generate(model=model, subject=subject, config=config)
        
        if ex_type == "qa":
            model.destroy() # Destroy the model to release memory
            del model
            model = None

        module_evaluate.evaluate_from_path(output_path, config)


if __name__ == "__main__":
    subjs, configuration, ip_server = prepare()
    run(subjs, configuration, ip_server)