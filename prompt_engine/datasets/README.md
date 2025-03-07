# Datasets

## Suported datasets
This directory hosts the datasets files. Currently supported and tested datasets are:

    - MMLU
    - MedMCQA
    - MedQA
    - PubmedQA
    - OpenMedQA (Open-Ended Question-Answering)


## Structure of the directories
Each dataset files are located in their own directory. Each dataset directory must contain at least two files for executing Medprompt, the validation/train examples and the test examples. However, to use SC-COT only the test file is needed.

The name of the files must be the same as the dataset folder, followed by "_val.json" or "_test.json". A example for the medqa 4 options would be:

    - medqa_4opt
        - "medqa_4opt_val.json"
        - "medqa_4opt_test.json"

If the dataset have multiple subjects, like MMLU, two files per subject are needed, following the same format. The name of the subject is added in the name of the files. Example:

    - mmlu
        - "mmlu_anatomy_val.json"
        - "mmlu_anatomy_test.json"
        - "mmlu_virology_val.json"
        - "mmlu_virology_test.json"
        ...

New datasets can be added to the project by adding new folders folllowing the structure of the project. If new datasets are added, make sure to include entries for them in the "medprompt/constants.py".


## Structure of the data
Each dataset must be a JSON with the following custom format:
```
{
    "ID1": {
        "question": "Which of the following terms describes the body's ability to maintain its normal state?",
        "correct_answer": "D",
        "options": {
        "A": "Option 1",
        "B": "Option 2",
        "C": "Option 3",
        "D": "Option 4"
        }
    },
    "ID2": {
        ...
    }
}
```
WARNING. The IDS of the questions must be unique.

If using an Open-Ended QA dataset like OpenMedqa, the "options" field is not necessary.
