import importlib
from run import prepare
from src.utils.constants import EXAMPLES_CONVERSION


if __name__ == "__main__":
    subjs, configuration, ip_server = prepare()

    ex_type = EXAMPLES_CONVERSION[configuration["config"]["dataset"]]
    if ex_type == "qa":
        module = importlib.import_module("src.qa.evaluate")
    else:
        module = importlib.import_module("src.medprompt.evaluate")
        
    for subject in subjs:
        module.evaluate_from_config(subject, configuration, ip_server)