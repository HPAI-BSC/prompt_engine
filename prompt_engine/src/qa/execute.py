from vllm import SamplingParams
from src.utils.constants import EXAMPLES_CONVERSION
from src.utils.prompt_templates import qa_datasets_prompts, qa_final_answer_prompt, reflexion_prompts
from src.utils.utils import *
from dataclasses import asdict
import random

from src.qa.reward_model import ArmoRM
from src.utils.logger import init_logger

logger = init_logger(__name__)


def generate_individual_answers(model, problems, selected_examples, n, output_path, sampling_params):
    # Generate prompts
    for p_id, problem in problems.items():
        system_prompt = qa_datasets_prompts["system_prompt"] if len(
            selected_examples[p_id]["solutions"]) > 0 else qa_datasets_prompts["zero_shot_system_prompt"]
        text = "\n\n".join([f"Question: {example['question']}\nAnswer: {example['answer']}" for example in selected_examples[p_id]["solutions"]]) + "\n\nQuestion:  " + problem["question"] + "\n\n"
        problem["prompt"] = model.tokenizer.apply_chat_template([{"role": "system", "content": system_prompt},
                                                                 {"role": "user", "content": text}],
                                                                 tokenize=False,
                                                                 add_generation_prompt=True,
                                                                 add_system_prompt=True).strip()
        # problem["prompt"] = model.generate_prompt(
        #     system_prompt, selected_examples[p_id]["solutions"], problem["question"], add_generation_prompt=True)

    # Generate answers
    possible_temperatures = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    possible_top_p = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print("Starting generation of individual answers...")
    for i in range(n):
        p = sampling_params.copy()
        
        if n > 1:
            p["temperature"] = random.choice(possible_temperatures)
            p["top_p"] = random.choice(possible_top_p)
            print(f"Generation {i+1} of {n}. Temperature: {str(p['temperature'])}\tTop P: {p['top_p']}")
            
        params = SamplingParams(**p)

        ids = list(problems.keys())
        prompts_text = [problem["prompt"] for problem in problems.values()]

        outputs = model.generate(prompts_text, params)
        logs = ""
        for i, output in enumerate(outputs):
            p_id = ids[i]
            metrics = asdict(output.metrics)
            generated_text = output.outputs[0].text.strip()
            r = {
                "response": generated_text,
                "prompt_tokens": len(output.prompt_token_ids),
                "generated_tokens": len(output.outputs[0].token_ids),
                "sampling_params": {"temperature": p["temperature"], "top_p": p["top_p"]},
                "metrics": metrics,
            }

            if "generations" not in problems[p_id]:
                problems[p_id]["generations"] = []
            problems[p_id]["generations"].append(r)

            if i % 100 == 0:
                logs += "############### Prompt for individual question {} ###############\n".format(
                    p_id)
                logs += prompts_text[i] + "\n"
                logs += "############### Response ###############\n"
                logs += generated_text + "\n\n"

    with open(os.path.join(output_path, "log.MD"), "a") as f:
        f.write(logs)
    return problems


def generate_final_answers(model, problems, output_path, sampling_params):
    system_prompt = qa_final_answer_prompt

    # Create prompts
    prompts = []
    for p_id, problem in problems.items():
        answers = [g["response"] for g in problem["generations"]]
        if len(answers) > 1:
            answers_str = "\n\n".join([f"Candidate answer {i+1}: {a}" for i, a in enumerate(answers)])
            
            text = f"Question: {problem['question']}\n\n" + answers_str
            
            text += "Please analyze each answer, extract the most relevant and accurate information, and combine it into a single, cohesive response. Ensure the resulting answer is clear, concise, and comprehensive. You can rephrase, reorganize, and refine the content as needed to create a superior answer. Think step by step.\nWrite the improved answer:"
            
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
        params = SamplingParams(**sampling_params)
        outputs = model.generate(prompts, params)

        for i, output in enumerate(outputs):
            p_id = list(problems.keys())[i]
            metrics = asdict(output.metrics)
            generated_text = output.outputs[0].text.strip()
            problems[p_id]["final_answer"] = {
                "prompt": prompts[i],
                "response": generated_text,
                "prompt_tokens": len(output.prompt_token_ids),
                "generated_tokens": len(output.outputs[0].token_ids),
                "metrics": metrics
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


def generate_reflection_answers(model, problems, selected_examples, n, output_path, config, sampling_params):
    params = SamplingParams(**sampling_params)
    ids = list(problems.keys())
    
    reward_model = ArmoRM(config["reward_model"], device_map="cuda:3")
    
    add_rewards = ['ultrafeedback-truthfulness','ultrafeedback-honesty','ultrafeedback-helpfulness', 'prometheus-score']
    
    for i in range(n):
        print(f"Iteration {i+1} of {n}...")
        # Generate answers
        if i == 0:
            prompts_text = []
            for p_id, problem in problems.items():
                system_prompt = qa_datasets_prompts["system_prompt"] if len(
                    selected_examples[p_id]["solutions"]) > 0 else qa_datasets_prompts["zero_shot_system_prompt"]
                p = model.generate_prompt(system_prompt, selected_examples[p_id]["solutions"], problem["question"], add_generation_prompt=True)
                prompts_text.append(p)
        else:
            prompts_text = [model.generate_prompt(reflexion_prompts["generate_answer"][-1], 
                                                  [],
                                                  "Question: {}\n\nAnswer: {}\n\nFeedback: {}\n\nReward attributes: {}\n\nInclude ONLY the improved response and nothing else.\n\nImproved response: ".format(problem["question"],
                                                                                                                               problem["generations"][-1]["response"], 
                                                                                                                               problem["generations"][-1]["feedback"],
                                                                                                                               str(problem["generations"][-1]["rewards"])),
                                                  add_generation_prompt=True)
                            for problem in problems.values()]
        
        print("Generating answers...")
        print(prompts_text[0])
        outputs = model.generate(prompts_text, params)
        logs = ""
        for i, output in enumerate(outputs):
            p_id = ids[i]
            metrics = asdict(output.metrics)
            generated_text = output.outputs[0].text.strip()
            
            messages = [
                { "role": "user", "content": problems[p_id]["question"]},
                { "role": "assistant", "content": generated_text}
            ]
            out_r = reward_model.get_str_rewards(messages)
            
            rewards = {key: out_r[key] for key in add_rewards}
            
            r = {
                "prompt": prompts_text[i],
                "response": generated_text,
                "prompt_tokens": len(output.prompt_token_ids),
                "generated_tokens": len(output.outputs[0].token_ids),
                "metrics": metrics,
                "rewards": rewards
            }

            if "generations" not in problems[p_id]:
                problems[p_id]["generations"] = []
            problems[p_id]["generations"].append(r)

            if i % 100 == 0:
                logs += "############### Prompt for individual question {} ###############\n".format(p_id)
                logs += prompts_text[i] + "\n"
                logs += "############### Response ###############\n"
                logs += generated_text + "\n\n"

        # Reflection
        feedback_prompts = [model.generate_prompt(reflexion_prompts["generate_feedback"][-1], 
                                                  [], 
                                                  "Question: {}\n\nAnswer: {}\n\nFeedback: ".format(problem["question"], 
                                                                                                    problem["generations"][-1]["response"]), 
                                                  add_generation_prompt=True) 
                             for problem in problems.values()]

        print("Generating feedbacks...")
        print(feedback_prompts[0])
        outputs = model.generate(feedback_prompts, params)
        for i, output in enumerate(outputs):
            p_id = ids[i]
            metrics = asdict(output.metrics)
            generated_text = output.outputs[0].text.strip()
            problems[p_id]["generations"][-1]["feedback_prompt"] = feedback_prompts[i]
            problems[p_id]["generations"][-1]["feedback"] = generated_text

            if i % 100 == 0:
                logs += "############### Prompt for feedback of question {} ###############\n".format(p_id)
                logs += feedback_prompts[i] + "\n"
                logs += "############### Response ###############\n"
                logs += generated_text + "\n\n"
    
    # Generate final answers
    prompts_text = [model.generate_prompt(reflexion_prompts["generate_answer"][-1], 
                                            [],
                                            "Question: {}\n\nAnswer: {}\n\nFeedback: {}\n\nReward attributes: {}\n\nInclude ONLY the final response and nothing else.\n\Final answer: ".format(problem["question"],
                                                                                                                        problem["generations"][-1]["response"], 
                                                                                                                        problem["generations"][-1]["feedback"],
                                                                                                                        str(problem["generations"][-1]["rewards"])),
                                            add_generation_prompt=True)
                    for problem in problems.values()]
    
    outputs = model.generate(prompts_text, params)
    for i, output in enumerate(outputs):
        p_id = ids[i]
        metrics = asdict(output.metrics)
        generated_text = output.outputs[0].text.strip()
        problems[p_id]["final_answer"] = {
            "prompt": prompts_text[i],
            "response": generated_text,
            "prompt_tokens": len(output.prompt_token_ids),
            "generated_tokens": len(output.outputs[0].token_ids),
            "metrics": metrics
        }
        if i % 100 == 0:
            logs += "############### Prompt for final answer generation of question {} ###############\n".format(
                p_id)
            logs += prompts_text[i] + "\n"
            logs += "############### Response ###############\n"
            logs += generated_text + "\n\n"
    
    del reward_model
    with open(os.path.join(output_path, "log.MD"), "a", encoding='utf8', errors="ignore") as f:
        f.write(logs)
    return problems


def execute(datastore, model, test_dataset, database_examples, out_test_path, configuration, sampling_params):
    dataset_filename = test_dataset[0]
    test_questions = test_dataset[1]
    test_questions = {k: test_questions[k]
                      for k in list(test_questions.keys())}

    if database_examples is None:
        examples = qa_datasets_prompts["examples"]
        selected_examples = {}
        for p_id in test_questions.keys():
            selected_examples[p_id] = {
                "solutions": examples.copy()[:configuration["k"]]
            }
    else:
        database_filename = database_examples[0]
        database_questions = database_examples[1]

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

    n_ensembles = int(configuration["ensembles"])

    if "final_answer" not in configuration or configuration["final_answer"] != "reflection":
        problems = generate_individual_answers(
            model, test_questions, selected_examples, n_ensembles, out_test_path, sampling_params)
        final_problems = generate_final_answers(
            model, problems, out_test_path, sampling_params)
    else:
        final_problems = generate_reflection_answers(
            model, test_questions, selected_examples, n_ensembles, out_test_path, configuration, sampling_params)

    with open(os.path.join(out_test_path, "generations.json"), "w") as f:
        json.dump(final_problems, f, indent=4)
