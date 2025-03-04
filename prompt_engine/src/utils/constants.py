import pathlib
import os

root = pathlib.Path(__file__).parent.parent.parent.resolve()    # Path of the root folder

DATASETS_PATH = os.path.join(root, "datasets")
DATABASES_PATH = os.path.join(root, "databases")
GENERATIONS_PATH = os.path.join(root, "outputs")

VALID_DATASETS = ["mmlu", "medmcqa", "medqa_4opt", "pubmedqa", "careqa", "openmedqa"]
VALID_SUBJECTS = ["anatomy", "clinical_knowledge", "college_biology", "college_medicine", "medical_genetics", "professional_medicine"]

VALID_RERANKERS = ["MedCPT-Cross-Encoder"]

EXAMPLES_CONVERSION = {
    "mmlu": "4_options",
    "medqa_4opt": "4_options",
    "medmcqa": "4_options",
    "careqa": "4_options",
    "pubmedqa": "yes/no/maybe",
    "openmedqa": "qa"
}