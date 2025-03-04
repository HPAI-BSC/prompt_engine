import os

from run import prepare
import importlib



def launch_evaluations(subjects, config, module):
    for subject in subjects:
        module.evaluate_from_config(subject, config)


if __name__ == "__main__":
    subjs, configuration, _ = prepare()

    ex_type = configuration["config"]["type"]
    
    if ex_type == "qa":
        module = importlib.import_module("src.qa.evaluate")
    else:
        module = importlib.import_module("src.medprompt.evaluate")
    launch_evaluations(subjs, configuration, module)


