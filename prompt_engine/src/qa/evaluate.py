import os
import json
import torch
import gc
import re
from src.utils.constants import GENERATIONS_PATH
from datetime import datetime

from src.utils.prompt_templates import qa_eval_prompt
from src.utils.openai_model import OpenAIModel
from src.utils.utils import load_json_file
from src.utils.logger import init_logger

logger = init_logger(__name__)


def multiple_llm_evaluation(problems, config, ip_server):
    """
        Evaluate the problems with the models specified in the configuration dictionary.

        Args:
            problems: dictionary of problems to evaluate
            config: configuration dictionary
            ip_server: IP address of the vLLM server
        
        Returns:
            problems: dictionary of problems with the evaluations
            result: dictionary with the evaluations
    """

    result = {"llm_evaluation": {}}
    if "evaluators" not in config:
        return None, None
    
    for i, c in enumerate(config["evaluators"]):
        logger.info(f"Evaluating with model: {c['model']}...")
        model = OpenAIModel(c, ip_server)

        logger.info(f"Starting LLM evaluation with model {model.model_name}...")
        problems, acc = llm_evaluation(model, problems)
        result["llm_evaluation"][model.model_name] = acc
        
        logger.info(f"Finished evaluation with model {model.model_name}.")
        model.destroy()
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Model destroyed.")

    return problems, result       


def llm_evaluation(evaluator, problems):
    """
        Evaluate the problems with the given evaluator model.

        Args:
            evaluator: evaluator model
            problems: dictionary of problems to evaluate

        Returns:
            problems: dictionary of problems with the evaluations
            acc: list of accuracies of the evaluations
    """
    
    ids = list(problems.keys())
    prompts = [
        evaluator.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": qa_eval_prompt},
                {
                    "role": "user",
                    "content": f"**Question**: {problem['question']}\n\n**Generated Answer**: {problem['final_answer']['response']}\n\n**Ground Truth**: {problem['correct_answer']}\n\n",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for problem in problems.values()
    ]
    
    samp_params = {"max_tokens": 2000 if "deepseek" in evaluator.model_name.lower() else 1500}
    model_responses = [None] * len(prompts)
    acc = []
    fails = 0
    for i in range(10):      
        logger.info(f"Evaluating {len(prompts)} generated responses... Try {i+1} of 10.") 
        outputs, _, _ = evaluator.generate(prompts, samp_params, batch_size=100)
        for j, response in enumerate(outputs):
            match = re.search(r"<verdict>(.+)</verdict>", response, re.IGNORECASE)
            if match:
                try:
                    score = match.group(1).strip()
                    score = int(score)
                    if score in [0, 1]:
                        if not "llm_score" in problems[ids[j]]["final_answer"]:
                            problems[ids[j]]["final_answer"]["llm_score"] = {}
                        problems[ids[j]]["final_answer"]["llm_score"][evaluator.model_name] = {
                            "score": score,
                            "response": response,
                            "prompt": prompts[j]
                        }
                        model_responses[j] = 1
                        acc.append(score)
                    else:
                        fails += 1
                        logger.warning(f"Not valid score in a match: {score}")
                except:
                    import traceback
                    traceback.print_exc()
                    logger.warning(f"Not valid score: {score}")
                    fails += 1
            else:
                fails += 1
                if fails % 5 == 0:
                    logger.info(f"Failed response example: {response}")

        ids = [p for z, p in enumerate(ids) if model_responses[z] == None]
        prompts = [p for z, p in enumerate(prompts) if model_responses[z] == None]
        model_responses = [None] * len(prompts)
        
        if len(prompts) == 0:
            logger.info("All problems evaluated correctly.")
            break
    return problems, acc
    

def evaluate_from_path(input_path, config, ip_server):
    """
        Evaluate the generations from the given path. It loads the problems from the generations.json file and evaluates them
        with the models specified in the configuration dictionary. The evaluations are saved in a new json file in the input path.

        Args:
            input_path: path to the generations file
            config: configuration dictionary
            ip_server: IP address of the vLLM server
    """

    # Load the problems
    problems = load_json_file(os.path.join(input_path, "generations.json"))

    # Evaluations
    problems, accuracies = multiple_llm_evaluation(problems, config, ip_server)
    if not accuracies:
        logger.info("No evaluations performed.")
        return
    
    reward_scores = {}

    # Format the LLM evaluations
    reward_scores["llm_evaluation"] = {}
    for key in accuracies["llm_evaluation"]:
        llm_score = sum(accuracies["llm_evaluation"][key]) / len(accuracies["llm_evaluation"][key])
        no_parsed = len(problems) - len(accuracies["llm_evaluation"][key])
        reward_scores["llm_evaluation"][key] = {
            "score": llm_score,
            "no_parsed": no_parsed
        }
    
    # Majority vote
    n_evaluators = len(accuracies["llm_evaluation"])
    majority_votes = []
    total_agreement = []
    for i, (key, p) in enumerate(problems.items()):
        evaluations = [accuracies["llm_evaluation"][model][i] for model in accuracies["llm_evaluation"] if i < len(accuracies["llm_evaluation"][model])]
        
        result = 1 if sum(evaluations) > n_evaluators / 2 else 0
        p["final_answer"]["llm_score"]["majority_voting"] = result
        majority_votes.append(result)
        
        agreement = 1 if sum(evaluations) == len(evaluations) else 0
        p["final_answer"]["llm_score"]["total_agreement"] = agreement
        total_agreement.append(agreement)
        
    reward_scores["llm_evaluation"]["majority_voting"] = sum(majority_votes) / len(majority_votes)
    reward_scores["llm_evaluation"]["total_agreement"] = sum(total_agreement) / len(majority_votes)

    logger.info(f"LLM evaluation scores: {reward_scores['llm_evaluation']}")
    logger.info(f"Evaluation saved in {input_path}.")
    with open(os.path.join(input_path, f"evaluation_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"), "w") as f:
        json.dump(reward_scores, f, indent=4)
        
    with open(os.path.join(input_path, f"scored_problems_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"), "w") as f2:
        json.dump(problems, f2, indent=4)



def evaluate_from_config(subject, config, ip_server):
    """
        Evaluate the generations from the configuration dictionary. 
        It builds the output path given the configuration.

        Args:
            subject: subject to evaluate
            config: configuration dictionary    
    """
    
    # Build the path to the generations file
    model_name = os.path.basename(config["vllm"]["model"]) if "vllm" in config else os.path.basename(config["openai"]["model"])

    final_answer_type = config["config"]["final_answer"] if "final_answer" in config["config"] else "merge"

    dataset_path = config["config"]["dataset"]
    if subject is not None:
        dataset_path = os.path.join(dataset_path, subject)

    embedding = os.path.join("QA", final_answer_type, os.path.basename(config["config"]["embedding"]))

    k_out = f"{config['config']['k']}k"
    if "database" in config["config"] and config["config"]["database"] is not None:
        k_out += f"_{config['config']['database']}"
    if "reranker" in config["config"] and config["config"]["reranker"] is not None:
        reranker_name = os.path.basename(config["config"]["reranker"]["path"])
        k_out += f"_{reranker_name}"

    if "working_dir" in config["config"]:
        working_dir = config["config"]["working_dir"]
    else:
        working_dir = GENERATIONS_PATH
        
    logger.info(f"Generations path: {working_dir}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Embedding {embedding}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"K: {k_out}")

    generations_path = os.path.join(working_dir, 
                                    "outputs",
                                    model_name, 
                                    embedding, 
                                    dataset_path,
                                    str(config["config"]["ensembles"]), 
                                    k_out)
    # Evaluate the generations
    evaluate_from_path(generations_path, config, ip_server)