import json
import gzip
import os
import random
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

from .logger import init_logger

logger = init_logger(__name__)
random.seed(8)

def filter_non_empty(dictionary):
    """
        Filter out None values from a dictionary.

        Args:
            dictionary: dictionary to filter

        Returns:
            dictionary: dictionary without None values    
    """

    return {key: value for key, value in dictionary.items() if value is not None}


def load_config(config_file):
    """
    Load configuration from a YAML file.
    
    Args:
        config_file: path to the configuration file

    Returns:
        config: dictionary with the configuration
    """

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def load_json_file(file_path):
    """
        Load a JSON file. It also supports GZ files.

        Args:
            file_path: path to the JSON file

        Returns:
            data: dictionary with the JSON file data    
    """

    logger.info(f"Loading {file_path}")
    if os.path.exists(file_path):
        if file_path.endswith(".gz"):
            with gzip.open(file_path, "rt") as f:
                data = json.load(f)
                logger.info("JSON from GZ loaded: " + str(len(data)) + " examples loaded.")
                return json.load(f)
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info("JSON loaded: " + str(len(data)) + " examples loaded.")
                return data
        else:
            logger.error(f"ERROR: Not a valid format file: {file_path}")
            raise ValueError(f"ERROR: Not a valid format file: {file_path}")
    else:
        logger.error(f"ERROR: File doesn't exists: {file_path}")
        raise FileNotFoundError(f"ERROR: File doesn't exists: {file_path}")


def save_json_file(file, data):
    """
    Save a JSON file.

    Args:
        file: path to the JSON file
        data: dictionary to save
    """

    with open(file, "w") as f:
        json.dump(data, f, indent=2)


def random_order(options, used_orders):
    """
        Generate a random order of the options.

        Args:
            options: list of options
            used_orders: list of used orders
        
        Returns:
            order: random order of the options
    """

    for i in range(10000):
        new_order = list(options)
        random.shuffle(new_order)
        order =  "".join(new_order)
        if order not in used_orders:
            return order
    return "".join(list(options))


def get_problem_description(problem, order):
    """
        Generate the description of the question with the defined order of the options.

        Args:
            problem: dictionary with the problem
            order: defined order of the options (str)
        
        Returns:
            description: description of the question (Question + ordered options)
    """
    description = ""

    # If PubMedQA, the context will be in the problem
    if "context" in problem:
        if type(problem["context"]) is list and len(problem["context"]) > 0:
            description = "Abstract: " + "\n".join(problem["context"])
        description += "\nQuestion: "

    description += problem["question"] + "\nOptions:\n"

    # Get the options in the defined order
    choices = problem["options"]
    options = list(choices.keys())
    reduced_order = ""
    for i, key in enumerate(order):
        if key not in choices:
            continue
        option = choices[key].strip(" \n")
        description += f"{options[i]}. {option}\n"
        reduced_order += key
    return description


def parse_response(problem, response, order):
    """ 
        Parse the response of the model and get the answer in the original order.

        Args:
            problem: dictionary with the problem
            response: response of the model
            order: defined order of the options (str)
        
        Returns:
            answer: answer in the original order
            original_answer: answer in the model order
    """

    # Split the response and search the answered option
    text = response.split("\nAnswer: ")[-1]
    original_answer = ""
    for option in order:    # Serch [A], [B]...
        if f"[{option}]" in text:   
            original_answer += option

    if len(original_answer) == 0:   # If the answer is not in the response
        return None, None
    
    if len(original_answer) > 1:    # If the answer is more than one character, get the last one
        original_answer = original_answer[-1]

    if original_answer not in list(problem["options"].keys()): # If the answer is not in the original options
        return None, None

    # Reorder
    problem_order = "".join(problem["options"].keys())  # Get original order
    answer = order[problem_order.find(original_answer)] # Get the answer in the original order

    return answer, original_answer


def load_solutions(path):
    """ 
        Load the correct solutions from a file. The file must be a problem file with the model solutions,
        or a database build witht the responses in the same format.

        Args:
            path: path to the generations.json file

        Returns:
            outputs: dictionary with the correct solutions
    """

    logger.info("Loading solutions...")
    if not path.endswith(".json"):
        path = os.path.join(path, "generations.json")
    problems = load_json_file(path)
    if problems is None:
        logger.error(f"ERROR: When trying to load solutions, the provided path doesn't exists: {path}")
        raise FileNotFoundError(f"ERROR: When trying to load solutions, the provided path doesn't exists: {path}")

    outputs = {}
    for p_id, problem in problems.items():
        correct_solutions = []
        if "generations" not in problem:
            continue
        for solution in problem["generations"]:
            # Load only correct solutions. If incorrect, skip
            if solution["result"] != problem["correct_answer"]:
                continue
            if "description" in solution:
                question = solution["description"]
            else:   # If database used, prompt won't appear in the solution. Generate the description of the question
                question = get_problem_description(problem, order=list(problem["options"].keys()))

            correct_solutions.append({"question": question, "answer": solution["response"].strip()})
        
        # Add only one correct solution for each problem
        if len(correct_solutions) > 0:
            outputs[p_id] = random.choice(correct_solutions)
    
    return outputs


def rerank_examples(problems, examples, config):
    """
        Rerank the examples using a reranker model.

        Args:
            problems: dictionary with the problems
            examples: dictionary with the previosly selected examples
            config: configuration dictionary
        
        Returns:
            selected_examples: dictionary with the selected examples
    """

    # Load reranker model
    tokenizer = AutoTokenizer.from_pretrained(config["reranker"]["path"])
    model = AutoModelForSequenceClassification.from_pretrained(config["reranker"]["path"])
    
    # Rerank the examples of each problem
    selected_examples = {}
    for test_id, problem in tqdm(problems.items(), desc="Reranking examples"):
        # Generate pairs of the question and the solution
        query = problem["question"]
        pairs = [[query, sol["question"]] for sol in examples[test_id]["solutions"]]

        # Get the scores of the pairs
        with torch.no_grad():
            encoded = tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            logits = model(**encoded).logits.squeeze(dim=1) # List of tensors with scores. Higher scores indicate higher relevance.

        # Sort the examples by the scores
        sorted_indices = torch.argsort(logits, descending=True).tolist()
        original_num_questions = config["k"]
        top_rerank = sorted_indices[:original_num_questions]

        # invert the order of the samples so that the most relevant is the most 'recent'
        top_rerank = top_rerank[::-1]

        # Save the selected examples
        selected_examples[test_id] = {
            "solutions": [],
            "scores": [],
            "examples": {
                "reranked": {
                    "ids": [],
                    "questions": []
                }, 
                "original": {
                    "ids": [],
                    "questions": []
                }
            }
        }

        selected_examples[test_id]["examples"]["original"]["ids"] = examples[test_id]["examples"]
        selected_examples[test_id]["examples"]["original"]["questions"] = examples[test_id]["solutions"]
        selected_examples[test_id]["examples"]["original"]["scores"] = examples[test_id]["scores"]


        for i in top_rerank:
            selected_examples[test_id]["solutions"].append(examples[test_id]["solutions"][i])
            selected_examples[test_id]["scores"].append(examples[test_id]["scores"][i])
            selected_examples[test_id]["examples"]["reranked"]["ids"].append(examples[test_id]["examples"][i])
            selected_examples[test_id]["examples"]["reranked"]["questions"].append(examples[test_id]["solutions"][i])

    return selected_examples



def load_qa_file(path):
    data = load_json_file(path)

    output = {}
    for key, value in data.items():
        output[key] = {}
        output[key]["id"] = value["id"]
        output[key]["question"] = value["question"]
        output[key]["correct_answer"] = value["options"][value["correct_answer"]] if "options" in value else value["correct_answer"]
    return output

def load_qa_database(path):
    data = load_json_file(path)

    output = {}
    for key, value in data.items():
        output[key] = {}
        output[key]["id"] = value["id"]
        output[key]["question"] = value["question"]
        if "options" in value:
            output[key]["correct_answer"] = value["options"][value["correct_answer"]]
        else:
            output[key]["correct_answer"] = value["correct_answer"]
        output[key]["answer"] = value["generations"][-1]["response"]
    return output