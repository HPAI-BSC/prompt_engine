import argparse
import os
import torch
import gc

from src.utils.model import Model
from src.utils.constants import *
from src.utils.utils import load_config, filter_non_empty

from src.generate import generate
from src.evaluate import evaluate_from_path

from src.utils.logger import init_logger

logger = init_logger(__name__)

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument(
        "configuration", type=str, help="Path of the YAML configuration file"
    )
    return p.parse_args()


# Check the format of the configuration file
def check_configuration(config):
    assert config["vllm"]['model'] is not None, "Model path must be specified"
    assert "dataset" in config["config"], "ERROR: Dataset not specified in configuration."
    assert config["config"]["dataset"] in VALID_DATASETS, "ERROR: Not a valid dataset: " + config["config"]["dataset"]
    assert "type" in config["config"], "ERROR: 'type' not specified in configuration. )"
    assert config["config"]["type"] in ["medprompt", "sc-cot"], "Not a valid execution type: " + config["config"]["type"]["name"] + ". Select one of the valid types (type: 'medprompt', 'sc-cot')."
    
    if config["config"]["type"] == "medprompt":
        assert "embedding" in config["config"]
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
    assert "vllm" in config, "ERROR: 'vllm' parameters not included in configuration."
    assert "config" in config, "ERROR: 'config' parameters not included in configuration."
    config["vllm"] = filter_non_empty(config["vllm"])
    config["config"] = filter_non_empty(config["config"])
    config["sampling_params"] = filter_non_empty(config["sampling_params"])

    check_configuration(config)

    subjects = format_dataset_subjects(config["config"])    # If mmlu, get all subjects to execute

    return subjects, config

def run(subjects, config):
    # Load the model using VLLM and config parameters
    model = Model(config["vllm"])
    
    logger.info("\nStarting " + config["config"]["type"].upper() + " with the following configuration:")
    logger.info(config)
    for subject in subjects:
        output_path = generate(model=model, subject=subject, config=config)
        evaluate_from_path(output_path, config)

    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    subjs, configuration = prepare()
    run(subjs, configuration)