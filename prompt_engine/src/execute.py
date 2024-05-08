import os
import traceback
import concurrent
from dataclasses import asdict
from vllm import SamplingParams
import time
import copy

from .utils.utils import *
from .utils.constants import EXAMPLES_CONVERSION
from .utils.prompt_templates import datasets_prompts

from .utils.logger import init_logger

logger = init_logger(__name__)


def infer(model, prompts, problems, sampling_params, n_retries, log_file):
    """
        Infer the responses for the prompts using the VLLM model

        Args:
            model: VLLM model
            prompts: list with the prompts
            problems: dictionary with the problems
            sampling_params: sampling parameters for the model
            n_retries: number of retries
            log_file: path to the log file
    """
    calls = 0
    model_responses = [None] * len(prompts)    # This will control retries for problems where answer can't be parsed
    
    for retry in range(n_retries):
        calls += 1
        params = copy.deepcopy(sampling_params)
        if "max_tokens" not in params:
            params["max_tokens"] = 500
        params["max_tokens"] += (100 * retry)
        
        if "temperature" not in params:
            params["temperature"] = 0.01
        params["temperature"] += (0.05 * retry)
        if params["temperature"] > 1:
            params["temperature"] = 1

        params = SamplingParams(**params)

        prompts_text = [p for _, _, p, _ in prompts] # Get only the text of the prompts
        outputs = model.generate(prompts_text, params)
        messages = []
        for i, output in enumerate(outputs):
            metrics = asdict(output.metrics)
            problem = problems[prompts[i][0]]
            order = prompts[i][1]

            generated_text = output.outputs[0].text.strip()
            result, original_answer = parse_response(problem, generated_text, order)

            message = f"########## Prompt ########## (id={prompts[i][0]} retry={retry+1})\n{prompts_text[i]}\n"
            message += f"########## Response ##########\n{generated_text}\n"
            if result is not None:
                output = {
                    "description": prompts[i][3],   # Get the original descrition of the question (Question + shuffled options)
                    "response": generated_text,
                    "original_answer": original_answer,
                    "result": result,
                    "order": order,
                    "retries": calls,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "generated_tokens": len(output.outputs[0].token_ids),
                    "metrics": metrics
                }
                if "generations" not in problem:
                    problem["generations"] = []
                problem["generations"].append(output)
                
                message += f"########## Answer ##########\nCorrect answer: {problem['correct_answer']} (Predicted answer: {result})\n"
                model_responses[i] = 1
            else:
                message += f"########## Invalid Answer ##########\nCorrect answer{problem['correct_answer']}\nPredicted answer:{generated_text}\n"
            messages.append(message)
        
        # Save logs each retry
        with open(log_file, "a") as f:
            for m in messages:
                f.write(m)
        
        save_json_file(log_file.replace("log.md", "intermediate_generations.json"), problems)

        # Set the prompts that have not been parsed to retry
        prompts = [p for j, p in enumerate(prompts) if model_responses[j] == None]
        model_responses = [None] * len(prompts)
        
        if len(prompts) == 0:
            print("All problems solved correctly.")
            return
        print("Model was not able to generate a correct answer for " + str(len(prompts)) +" prompts. Try " + str(retry+1) + " of " + str(n_retries))


def generate_problem_prompts(model, problem_id, problem, selected_examples, system_prompt, configuration):
    """
        Generate the prompts for a problem

        Args:
            model: VLLM model
            problem_id: id of the problem
            problem: dictionary with the problem
            selected_examples: dictionary with the selected examples
            system_prompt: chat template of the model
            configuration: configuration dictionary

        Returns:
            prompts: list with the prompts
    """
    prompts = []
    used_order = []
    if "scores" in selected_examples:
        problem["scores"] = selected_examples["scores"]
    if "examples" in selected_examples:
        problem["examples"] = selected_examples["examples"]

    for n in range(configuration["ensembles"]):
        if configuration["shuffle"]:
            order = random_order("".join(problem["options"].keys()), used_order)
            used_order.append(order)
        else:
            order = "".join(problem["options"].keys())
        description = get_problem_description(problem, order)
        
        # Generate prompt using the chat template of the model
        p = model.generate_prompt(system_prompt, selected_examples["solutions"], description, add_generation_prompt=True)
        prompts.append((problem_id, order, p, description))
    return prompts


def generate_prompts(datastore, model, problems, solutions, problems_filename, database_filename, knn, configuration):
    """
        Generate the prompts for the problems in the dataset

        Args:
            datastore: database object
            model: VLLM model
            problems: dictionary with the problems
            solutions: dictionary with the solutions
            problems_filename: name of the dataset file
            database_filename: name of the database file
            knn: boolean to know if knn is used
            configuration: configuration dictionary

        Returns:
            prompts: list with the prompts
            times: dictionary with the times of the different steps
    """
    select_time = None
    rerank_time = None

    if knn:
        start_select_time = time.time()
        logger.info("Selecting KNN examples for the prompts")
        selected_examples = datastore.select_examples(problems, solutions, problems_filename, database_filename)   # If knn, search the most similar examples into the generated database examples
        for key, output in selected_examples.items():
            selected_examples[key] = {
                "solutions": [solutions[i] for i in output["ids"] if i in solutions],
                "scores": output["scores"],
                "examples": output["examples"]
            }
        select_time = time.time() - start_select_time
        
        if "reranker" in configuration and configuration["reranker"]:
            logger.info("Reranking examples...")
            start_rerank_time = time.time()
            selected_examples = rerank_examples(problems, selected_examples, configuration)
            rerank_time = time.time() - start_rerank_time

    else:
        # If not knn, select the first K static examples
        selected_examples = {}
        for p_id in problems.keys():
            selected_examples[p_id] = {
                "solutions": solutions.copy()[:configuration["k"]]
                }
    
    # With the selected examples for each problem, generate the prompts
    logger.info("Examples prepared. Generating individual prompts for each ensemble.")
    system_prompt = datasets_prompts[EXAMPLES_CONVERSION[configuration["dataset"]]]["system_prompt"]
    prompts = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for each problem
        futures = {
            executor.submit(
                generate_problem_prompts, 
                model=model,
                problem_id=problem_id, 
                problem=problem,
                selected_examples=selected_examples[problem_id],
                system_prompt=system_prompt, 
                configuration=configuration): 
            problem_id for problem_id, problem in problems.items()
        }

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            for p_id, order, prompt, description in future.result():
                prompts.append((p_id, order, prompt, description))
                
    times = {}
    if select_time:
        times["knn_selection"] = select_time
    if rerank_time:
        times["rerank_time"] = rerank_time
    return prompts, times


def execute(datastore, model, dataset_path, output_path, examples, database_filename, configuration, sampling_params):
    """
        Execute the pipeline to generate the responses for the problems in the dataset

        Args:
            datastore: database object
            model: VLLM model
            dataset_path: path to the dataset
            output_path: path to save the results
            examples: path to the examples database
            database_filename: path to the database
            configuration: configuration dictionary
            sampling_params: sampling parameters for the model

        Returns:
            times: dictionary with the times of the different steps
    """

    times = {}

    # Load problems
    start_load_problems = time.time()
    try:
        problems = load_json_file(dataset_path)
    except:
        logger.error(f"Error loading the dataset {dataset_path}")
        return
    
    times["load_problems"] = time.time() - start_load_problems

    # Set the log filename and create output fodlers structure
    log_file = os.path.join(output_path, "log.md")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    
    # Load examples
    if examples is None:
        # If examples is None, use the default few-shot examples defined in prompt_templates.py
        knn = False 
        solutions = datasets_prompts[EXAMPLES_CONVERSION[configuration["dataset"]]]["examples"]
    else:
        # If examples is not None, means that we have to use examples from the database as few-shots
        knn = True

        # Load the solutions of the database examples
        start_load_examples = time.time()
        solutions = load_solutions(examples)
        times["load_examples"] = time.time() - start_load_examples
    
    try:
        # Generate the prompts of all the problems
        start_generate_prompts = time.time()
        logger.info("Generating prompts...")
        prompts, select_time = generate_prompts(datastore=datastore, 
                                                model=model, 
                                                problems=problems,
                                                solutions=solutions,
                                                problems_filename=os.path.basename(dataset_path),
                                                database_filename=database_filename,
                                                knn=knn,
                                                configuration=configuration)

        times["generate_prompts"] = {"total": time.time() - start_generate_prompts}
        if select_time:
            for key, value in select_time.items():
                times["generate_prompts"][key] = value
        
        if "val" in dataset_path:
            n_retries = 3
        else:
            n_retries = 10
            
        # Infer the prompts using VLLM
        start_infer = time.time()
        infer(model, prompts=prompts, problems=problems, sampling_params=sampling_params, n_retries=n_retries, log_file=log_file)
        times["llm_infer"] = time.time() - start_infer

        # Save the results into a JSON file
        start_save = time.time()
        logger.info(f"Saving generation results to {output_path}")
        save_json_file(os.path.join(output_path, "generations.json"), problems)
        times["save_results"] = time.time() - start_save
        return times
    except KeyboardInterrupt:
        quit()
    except:
        traceback.print_exc()