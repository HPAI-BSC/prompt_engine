from vllm import SamplingParams
from src.utils.constants import GENERATIONS_PATH
from src.utils.prompt_templates import datasets_prompts, reflexion_prompts
from src.utils.utils import *
from dataclasses import asdict
import random

from src.utils.logger import init_logger

logger = init_logger(__name__)


def generate_merge_individual_answers(model, problems, selected_examples, n, output_path, sampling_params):
    # Generate prompts
    for p_id, problem in problems.items():
        system_prompt = datasets_prompts["qa"]["system_prompt"] if len(
            selected_examples[p_id]["solutions"]) > 0 else datasets_prompts["qa"]["zero_shot_system_prompt"]

        #pr = model.generate_prompt(system_prompt, selected_examples[p_id]["solutions"], problem["question"], add_generation_prompt=True)
        pr = model.generate_prompt_single_turn(system_prompt, selected_examples[p_id]["solutions"], problem["question"], add_generation_prompt=True)
        problem["prompt"] = pr

    # Generate answers
    possible_temperatures = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    possible_top_p = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    logger.info("Starting generation of individual answers...")
    for i in range(n):
        p = sampling_params.copy()
        
        if n > 1:
            p["temperature"] = random.choice(possible_temperatures)
            p["top_p"] = random.choice(possible_top_p)
            logger.info(f"Generation {i+1} of {n}. Temperature: {str(p['temperature'])}\tTop P: {p['top_p']}")

        ids = list(problems.keys())
        prompts_text = [problem["prompt"] for problem in problems.values()]

        outputs, tokens, metrics = model.generate(prompts_text, p)
        logs = ""
        for i, generated_text in enumerate(outputs):
            p_id = ids[i]
            r = {
                "prompt": prompts_text[i],
                "response": generated_text,
                "prompt_tokens": tokens[i]["prompt"] if tokens[i] is not None else 0,
                "generated_tokens": tokens[i]["generated"] if tokens[i] is not None else 0,
                "total_tokens": tokens[i]["total"] if tokens[i] is not None else 0,
                "metrics": metrics[i]
            }

            if "generations" not in problems[p_id]:
                problems[p_id]["generations"] = []
            problems[p_id]["generations"].append(r)

            if i % 100 == 0:
                logs += "############### Prompt for individual question {} ###############\n".format(
                    p_id)
                logs += str(prompts_text[i]) + "\n"
                logs += "############### Response ###############\n"
                logs += generated_text + "\n\n"

    with open(os.path.join(output_path, "log.MD"), "a") as f:
        f.write(logs)
    return problems


def generate_merge_final_answers(model, problems, output_path, sampling_params):
    system_prompt = datasets_prompts["qa"]["merge_answers_system_prompt"]

    # Create prompts
    prompts = []
    for p_id, problem in problems.items():
        answers = [g["response"] for g in problem["generations"]]
        if len(answers) > 1:
            answers_str = "\n\n".join([f"Candidate answer {i+1}: {a}" for i, a in enumerate(answers)])
            
            text = f"Question: {problem['question']}\n\n" + answers_str
            
            text += datasets_prompts["qa"]["merge_answers_final_instruction"]
            
            p = model.tokenizer.apply_chat_template([{"role": "system", "content": system_prompt},
                                                     {"role": "user", "content": text}],
                                                     tokenize=False,
                                                     add_generation_prompt=True,
                                                     add_system_prompt=True).strip()
            prompts.append(p)

    logs = ""
    if len(prompts) > 0:
        # Generate final answers
        print("Generating final answers...")
        outputs, tokens, metrics = model.generate(prompts, sampling_params)

        for i, generated_text in enumerate(outputs):
            p_id = list(problems.keys())[i]
            problems[p_id]["final_answer"] = {
                "prompt": prompts[i],
                "response": generated_text,
                "prompt_tokens": tokens[i]["prompt"] if tokens[i] is not None else 0,
                "generated_tokens": tokens[i]["generated"] if tokens[i] is not None else 0,
                "total_tokens": tokens[i]["total"] if tokens[i] is not None else 0,
                "metrics": metrics[i]
            }
            if i % 100 == 0:
                logs += "############### Prompt for final answer generation of question {} ###############\n".format(
                    p_id)
                logs += prompts[i] + "\n"
                logs += "############### Response ###############\n"
                logs += generated_text + "\n\n"
    else:
        for p_id, problem in problems.items():
            problems[p_id]["final_answer"] = {
                "response": problem["generations"][0]["response"],
                "prompt": "",
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "metrics": {}
            }
    with open(os.path.join(output_path, "log.MD"), "a") as f:
        f.write(logs)
    return problems

def build_prompts_reflection(model, problems, config):
    if "database" in config and "deepseek" in config["database"].lower():
        final_note = "Remember: Start with your internal reasoning in <think> ... </think> tokens, then provide your final revised answer."
        system_prompt = reflexion_prompts["generate_answer_thinking"][-1]
    else:
        final_note = "Provide only the improved response."
        system_prompt = reflexion_prompts["generate_answer"][-1]

    return [
        model.generate_prompt(
            system_prompt, 
            [],
            f"Question: {problem['question']}\n\nPrevious Answer: {problem['generations'][-1]['response']}\n\nFeedback: {problem['generations'][-1]['feedback']}\n\n{final_note}\n",
            add_generation_prompt=True
        ) 
        for problem in problems.values()
    ]

def build_prompts_feedback(model, problems):
    return [
        model.generate_prompt(
            reflexion_prompts["generate_feedback"][-1], 
            [], 
            f"Question: {problem['question']}\n\nAnswer: {problem['generations'][-1]['response']}\n\nYour Feedback: ",
            add_generation_prompt=True
        ) 
        for problem in problems.values()
    ]

def generate_reflection_answers(model, problems, selected_examples, n, output_path, config, sampling_params):
    ids = list(problems.keys())
    
    for i in range(n):
        print(f"Iteration {i+1} of {n}...")
        # Generate answers
        if i == 0:
            prompts_text = []
            for p_id, problem in problems.items():
                system_prompt = datasets_prompts["qa"]["system_prompt"] if len(
                    selected_examples[p_id]["solutions"]) > 0 else datasets_prompts["qa"]["zero_shot_system_prompt"]
                p =  model.generate_prompt_single_turn(system_prompt, selected_examples[p_id]["solutions"], problem["question"], add_generation_prompt=True)
                prompts_text.append(p)
        else:
            prompts_text = build_prompts_reflection(model, problems, config)
        
        logger.info(f"Example of prompt in iteration {i+1}: {prompts_text[0]}")
        
        logger.info("Generating answers...")
        outputs, tokens, metrics = model.generate(prompts_text, sampling_params)
        logs = ""
        for i, generated_text in enumerate(outputs):
            p_id = ids[i]      
            r = {
                "prompt": prompts_text[i],
                "response": generated_text,
                "prompt_tokens": tokens[i]["prompt"] if tokens[i] is not None else 0,
                "generated_tokens": tokens[i]["generated"] if tokens[i] is not None else 0,
                "total_tokens": tokens[i]["total"] if tokens[i] is not None else 0,
                "metrics": metrics[i],
            }

            if "generations" not in problems[p_id]:
                problems[p_id]["generations"] = []
            problems[p_id]["generations"].append(r)

            if i % 100 == 0:
                logs += "############### Prompt for individual question {} ###############\n".format(p_id)
                logs += prompts_text[i] + "\n"
                logs += "############### Response ###############\n"
                logs += generated_text + "\n\n"

        # Generate feedback
        feedback_prompts = build_prompts_feedback(model, problems)
        logger.info(f"Example of feedback prompt in iteration {i+1}: {feedback_prompts[0]}")
        logger.info("Generating feedbacks...")
        outputs, _, _ = model.generate(feedback_prompts, sampling_params)
        for i, generated_text in enumerate(outputs):
            p_id = ids[i]
            problems[p_id]["generations"][-1]["feedback_prompt"] = feedback_prompts[i]
            problems[p_id]["generations"][-1]["feedback"] = generated_text

            if i % 100 == 0:
                logs += "############### Prompt for feedback of question {} ###############\n".format(p_id)
                logs += feedback_prompts[i] + "\n"
                logs += "############### Response ###############\n"
                logs += generated_text + "\n\n"
    
    # Generate final answers
    prompts_text = build_prompts_reflection(model, problems, config)
    
    outputs, tokens, metrics = model.generate(prompts_text, sampling_params)
    for i, generated_text in enumerate(outputs):
        p_id = ids[i]
        problems[p_id]["final_answer"] = {
                "prompt": prompts_text[i],
                "response": generated_text,
                "prompt_tokens": tokens[i]["prompt"] if tokens[i] is not None else 0,
                "generated_tokens": tokens[i]["generated"] if tokens[i] is not None else 0,
                "total_tokens": tokens[i]["total"] if tokens[i] is not None else 0,
                "metrics": metrics[i]
            }
        if i % 100 == 0:
            logs += "############### Prompt for final answer generation of question {} ###############\n".format(
                p_id)
            logs += prompts_text[i] + "\n"
            logs += "############### Response ###############\n"
            logs += generated_text + "\n\n"
    
    with open(os.path.join(output_path, "log.MD"), "a", encoding='utf8', errors="ignore") as f:
        f.write(logs)
    return problems


def execute(datastore, model, test_dataset, database_examples, out_test_path, configuration, sampling_params):
    """
        Execute the generation process with the given parameters. It generates the final answers for the given test dataset
        using the model and the database examples. The final answers are saved in a json file in the output path.

        Args:
            datastore: database client
            model: model to use for generation
            test_dataset: tuple with the test dataset filename and the questions
            database_examples: tuple with the database examples filename and the questions
            out_test_path: output path to save the final answers
            configuration: configuration dictionary
            sampling_params: sampling parameters
    """
    dataset_filename = test_dataset[0]
    test_questions = test_dataset[1]
    test_questions = {k: test_questions[k] for k in list(test_questions.keys())}

    if database_examples is None:
        examples = datasets_prompts["qa"]["examples"]
        selected_examples = {}
        for p_id in test_questions.keys():
            selected_examples[p_id] = {
                "solutions": examples.copy()[:configuration["k"]] if "k" in configuration and configuration["k"] > 0 else [],
            }
    else:
        database_filename = database_examples[0]
        database_questions = database_examples[1]

        if datastore:
            selected_examples = datastore.select_examples(problems=test_questions,
                                                        solutions=database_questions,
                                                        problems_collection=dataset_filename,
                                                        database_path=database_filename)
            for key, output in selected_examples.items():
                selected_examples[key] = {
                    "solutions": [database_questions[i] for i in output["ids"] if i in database_questions],
                    "scores": output["scores"],
                    "examples": output["examples"]
                }
            if "reranker" in configuration and configuration["reranker"]:
                logger.info("Reranking examples...")
                selected_examples = rerank_examples(
                    database_questions, selected_examples, configuration)
        else:
            # If not datastore, use K static or random examples
            logger.info(f"No datastore. Using {configuration['k']} static examples. {configuration['embedding']} strategy.")
            if "embedding" in configuration and configuration["embedding"] == "random":
                static_examples = random.sample(list(database_questions.values()), configuration["k"])
            else:
                static_examples = database_questions[:configuration["k"]] if isinstance(database_questions, list) else list(database_questions.values())[:configuration["k"]]
            selected_examples = {p_id: {"solutions": static_examples} for p_id in test_questions.keys()}

    n_ensembles = int(configuration["ensembles"])
    
    if "final_answer" not in configuration or configuration["final_answer"] != "reflection":
        problems = generate_merge_individual_answers(
            model, test_questions, selected_examples, n_ensembles, out_test_path, sampling_params)
        final_problems = generate_merge_final_answers(
            model, problems, out_test_path, sampling_params)
    else:
        final_problems = generate_reflection_answers(
            model, test_questions, selected_examples, n_ensembles, out_test_path, configuration, sampling_params)

    with open(os.path.join(out_test_path, "generations.json"), "w") as f:
        json.dump(final_problems, f, indent=4)
