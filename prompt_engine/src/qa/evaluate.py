import os
import json
import torch
import gc
from vllm import SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from src.utils.constants import GENERATIONS_PATH


from src.utils.prompt_templates import qa_eval_prompt
from src.qa.reward_model import ArmoRM
from src.utils.model import Model


def llm_evaluation(problems, config):
        
    evaluator = Model(config["evaluator"])
    
    system_prompt = qa_eval_prompt
        
    ids = list(problems.keys())
    prompts = [
        evaluator.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Generated Answer: {problem['final_answer']['response']}\nGround Truth: {problem['correct_answer']}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for problem in problems.values()
    ]
    
    model_responses = [None] * len(prompts)
    acc = []
    for i in range(5):      
        print(f"Evaluating {len(prompts)} generated responses... Try {i+1} of 5.") 
        outputs = evaluator.generate(prompts, SamplingParams(**config["sampling_params"]))
        for j, output in enumerate(outputs):
            score = output.outputs[0].text.strip()
            print(score)
            try:
                score = int(score[0])
                if score in [0, 1]:
                    problems[ids[j]]["final_answer"]["llm_score"] = score
                    model_responses[j] = 1
                    acc.append(score)
            except:
                print("Not valid score")
                pass

        ids = [p for j, p in enumerate(ids) if model_responses[j] == None]
        prompts = [p for j, p in enumerate(prompts) if model_responses[j] == None]
        model_responses = [None] * len(prompts)
        
        if len(prompts) == 0:
            print("All problems evaluated correctly.")
            break
    
    destroy_model_parallel()
    destroy_distributed_environment()
    del evaluator.model.llm_engine.model_executor
    del evaluator
    torch.cuda.empty_cache()
    gc.collect()

    return problems, acc

def reward_model_evaluation(problems, config):
        
    model = ArmoRM(config["config"]["reward_model"], device_map="cuda:2")
    
    scores = []
    rewards = []
    for problem in problems.values():
        messages = [
            { "role": "user", "content": problem["question"]},
            { "role": "assistant", "content": problem["final_answer"]["response"]}
        ]
        score, multi_obj_rewards = model(messages)
        problem["final_answer"]["score"] = score
        
        r = {}
        for i in range(len(multi_obj_rewards)):
            r[model.attributes[i]] = multi_obj_rewards[i]
        problem["final_answer"]["multi_obj_rewards"] = r
        scores.append(score)
        rewards.append(multi_obj_rewards)
    
    final_scores = {
        "preference_score": sum(scores) / len(scores)
    }

    for key in range(len(rewards[0])):
        final_scores[model.attributes[key]] = sum([r[key] for r in rewards]) / len(rewards)
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return problems, final_scores
    
    

def evaluate_from_path(input_path, config):
    """
    Evaluate the generated answers using the reward model
    """

    # Load the problems
    with open(os.path.join(input_path, "generations.json"), "r") as f:
        problems = json.load(f)

    problems, accuracies = llm_evaluation(problems, config)
    problems, reward_scores = reward_model_evaluation(problems, config)
    
    llm_score = sum(accuracies) / len(accuracies) if len(accuracies) > 0 else 0
    no_parsed = len(problems) - len(accuracies)
    reward_scores["llm_evaluation"] = {
        "accuracy": llm_score,
        "no_parsed_evaluations": no_parsed
    }
    print(reward_scores)
    
    with open(os.path.join(input_path, "evaluation.json"), "w") as f:
        json.dump(reward_scores, f, indent=4)
        
    with open(os.path.join(input_path, "scored_problems.json"), "w") as f2:
        json.dump(problems, f2, indent=4)



def evaluate_from_config(subject, config):
    """
        Evaluate the generations from the configuration dictionary. 
        It builds the output path given the configuration.

        Args:
            subject: subject to evaluate
            config: configuration dictionary    
    """
    
    # Build the path to the generations file
    model_name = os.path.basename(config["vllm"]["model"])
    ex_type = config["config"]["type"]
    final_answer_type = config["config"]["final_answer"]
    dataset_path = config["config"]["dataset"]
    if subject is not None:
        dataset_path = os.path.join(dataset_path, subject)

    if ex_type == "medprompt":
        embedding = os.path.basename(config["config"]["embedding"])
    elif ex_type == "qa":
        embedding = os.path.join("QA", config["config"]["final_answer"], os.path.basename(config["config"]["embedding"]))
    else:
        embedding = "SC-COT"

    k_out = f"{config['config']['k']}k"
    if "database" in config["config"] and config["config"]["database"] is not None:
        k_out += f"_{config['config']['database']}"
    if "reranker" in config["config"] and config["config"]["reranker"] is not None:
        reranker_name = os.path.basename(config["config"]["reranker"]["path"])
        k_out += f"_{reranker_name}"

    print("Generations path: ", GENERATIONS_PATH)
    print("Model: ", model_name)
    print("Embedding", embedding)
    print("Dataset:", dataset_path)
    print("K: ", k_out)

    generations_path = os.path.join(GENERATIONS_PATH, 
                                    model_name, 
                                    embedding, 
                                    dataset_path,
                                    str(config["config"]["ensembles"]), 
                                    k_out)
    print(generations_path)
    # Evaluate the generations
    evaluate_from_path(generations_path, config)