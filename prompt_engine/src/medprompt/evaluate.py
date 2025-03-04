import os
import json
import sklearn
import sklearn.metrics
import datetime, pytz
import numpy as np

from src.utils.utils import load_json_file, save_json_file, filter_non_empty
from src.utils.constants import GENERATIONS_PATH

def evaluate(file_path, config):
    """
        Evaluate a generation file
        
        Args: 
            file_path: path to the generations file
            config: configuration dictionary

        Returns:
            output: evaluation dictionary         
    """

    # Load the generations file
    data = load_json_file(file_path)
    
    # Get the generation date (modification date of the generations file)
    try:
        m_time = os.path.getmtime(file_path)
        generation_date = datetime.datetime.fromtimestamp(m_time, pytz.timezone('Europe/Madrid')).strftime('%d-%m-%Y %H:%M:%S')
    except:
        m_time = None

    if config and "config" in config:
        config["config"] = filter_non_empty(config["config"])   # Some config parameters could be modified by generate() to null

    # Initialize the evaluation variables
    y_true = []
    y_pred = []
    answer_counts = []
    skipped = 0

    # Prompt lenght statistics
    total_generated_tokens = 0
    total_input_tokens = 0
    total_inference_time = 0
    mean_prompt = []
    max_prompt = []
    mean_generation = []
    max_generation = []
    mean_total_length = []
    max_total_length = []

    options = {}    # A,B,C,D options statistics
    draws = {"total": 0, "correct": 0, "incorrect": 0}  # Draw statistics

    scores = []

    incorrect_questions = {}
    # Evaluate the generations
    for problem_id, problem in data.items():
        voting = {}
        prompt_length = []
        generated_length = []
        total_length = []
        if "generations" in problem:
            for response in problem["generations"]:
                if response["result"] in voting:
                    voting[response["result"]] += 1
                else:
                    voting[response["result"]] = 1

                if "original_answer" in response:
                    if response["original_answer"] in options:
                        options[response["original_answer"]] += 1
                    else:
                        options[response["original_answer"]] = 1
                total_generated_tokens += response["tokens"]["generated"]
                total_input_tokens += response["tokens"]["prompt"]
                if "metrics" in response and response["metrics"] is not None:
                    inference_time = response["metrics"]["finished_time"] - response["metrics"]["first_scheduled_time"]
                    total_inference_time += inference_time
                prompt_length.append(response["tokens"]["prompt"])
                generated_length.append(response["tokens"]["generated"])
                total_length.append(int(response["tokens"]["prompt"]) + int(response["tokens"]["generated"]))

            if len(prompt_length) > 0:
                mean_prompt.append(sum(prompt_length) / len(prompt_length))
                max_prompt.append(max(prompt_length))
            if len(generated_length) > 0:
                mean_generation.append(sum(generated_length) / len(generated_length))
                max_generation.append(max(generated_length))
            if len(total_length) > 0:
                mean_total_length.append(sum(total_length) / len(total_length))
                max_total_length.append(sum(total_length) / len(total_length))

        # Majority voting
        best_answer = ""
        best_count = 0
        for k, v in voting.items():
            if v > best_count:
                best_answer = k
                best_count = v
        if not best_answer:
            skipped += 1
            best_answer = "NULL"    # If not able of parsing any answer from CoT. Count as incorrect
        
        y_true.append(problem["correct_answer"])
        answer_counts.append(len(voting))
        y_pred.append(best_answer)

        # If incorrect answer
        if problem["correct_answer"] != best_answer:
            incorrect_questions[problem_id] = problem
            incorrect_questions[problem_id]["voting"] = best_answer

        # If draw, add statistics
        if any(v == best_count and k != best_answer for k, v in voting.items()):
            draws["total"] += 1
            if best_answer == problem["correct_answer"]:
                draws["correct"] += 1
            else:
                draws["incorrect"] += 1
        
        # Add cosine distance mean
        if "scores" in problem:
            scores.append(np.mean(problem["scores"]))
    
    output = {
        "count": len(y_true),
        "accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
        "no_parsed_answers": skipped,
        "mean_different_answers": sum(answer_counts) / len(answer_counts),
        "draws": draws,
        "options": options,
        "mean_cosine_similarity": np.mean(scores) if len(scores) > 0 else 0,
        "statistics": {
            "total_generated_tokens": total_generated_tokens,
            "total_input_tokens": total_input_tokens,
            "total_inference_time": total_inference_time,
            "tokens/s": total_generated_tokens / total_inference_time if total_inference_time > 0 else 0,
            "max_prompt_length": max(max_prompt),
            "mean_prompt_length": sum(mean_prompt) / len(mean_prompt),
            "max_generation_length": max(max_generation),
            "mean_generated_length": sum(mean_generation) / len(mean_generation),
            "max_total_length": max(max_total_length),
            "mean_total_length":  sum(mean_total_length) / len(mean_total_length),
        },
        "incorrect_questions": incorrect_questions,
        "config": config,
        "generation_date": generation_date,
        "date": datetime.datetime.now(pytz.timezone('Europe/Madrid')).strftime('%d-%m-%Y %H:%M:%S') 
    }
    return output


def evaluate_from_path(path, config):
    """
        Evaluate the generations from the output path

        Args:
            path: path to the generations file
            config: configuration dictionary
    """

    output = evaluate(os.path.join(path, "generations.json"), config)
    filename = f"evaluation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"

    # Save the incorrect questions in a separate file
    incorrect_questions = output["incorrect_questions"]
    save_json_file(os.path.join(path, filename.replace("evaluation_", "incorrect_questions_")), incorrect_questions)
    del output["incorrect_questions"]

    print(json.dumps(output, indent=2))
    save_json_file(os.path.join(path, filename), output)


def evaluate_from_config(subject, config):
    """
        Evaluate the generations from the configuration dictionary. 
        It builds the output path given the configuration.

        Args:
            subject: subject to evaluate
            config: configuration dictionary    
    """
    
    # Build the path to the generations file
    model_name = os.path.basename(config["vllm"]["model"]) if "vllm" in config else os.path.basename(config["openai"]["model"])
    ex_type = config["config"]["type"]
    dataset_path = config["config"]["dataset"]
    if subject is not None:
        dataset_path = os.path.join(dataset_path, subject)

    if ex_type == "medprompt":
        embedding = os.path.basename(config["config"]["embedding"])
    else:
        embedding = "SC-COT"

    k_out = f"{config['config']['k']}k"
    if "database" in config["config"] and config["config"]["database"] is not None:
        k_out += f"_{config['config']['database']}"
    if "reranker" in config["config"] and config["config"]["reranker"] is not None:
        reranker_name = os.path.basename(config["config"]["reranker"]["path"])
        k_out += f"_{reranker_name}"

    working_dir = config["config"]["working_dir"] if "working_dir" in config["config"] else GENERATIONS_PATH
    generations_path = os.path.join(working_dir, 
                                    "outputs",
                                    model_name, 
                                    embedding, 
                                    dataset_path,
                                    str(config["config"]["ensembles"]), 
                                    k_out)
    
    # Evaluate the generations
    evaluate_from_path(generations_path, config)