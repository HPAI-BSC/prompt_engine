from src.utils.prompt_templates import datasets_prompts
from src.utils.utils import *
import random

from src.utils.logger import init_logger

logger = init_logger(__name__)


def generate_answers(model, problems, selected_examples, output_path, sampling_params):
    """
        Generate the final answers for the given problems using the model and the selected examples. The final answers are saved in a json file in the output path.

        Args:
            model: model to use for generation
            problems: dictionary of problems to generate the final answers
            selected_examples: dictionary of selected examples to use for generation
            output_path: output path to save the final answers
            sampling_params: sampling parameters

        Returns:
            problems: dictionary of problems with the final answers
    """
    # Generate prompts
    for p_id, problem in problems.items():
        system_prompt = datasets_prompts["qa"]["system_prompt"] if len(selected_examples[p_id]["solutions"]) > 0 else datasets_prompts["qa"]["zero_shot_system_prompt"]
        pr = model.generate_prompt_single_turn(system_prompt, selected_examples[p_id]["solutions"], problem["question"], add_generation_prompt=True)
        problem["prompt"] = pr

    ids = list(problems.keys())
    prompts_text = [problem["prompt"] for problem in problems.values()]

    logger.info(f"Starting generation of {len(prompts_text)} prompts...")
    outputs, tokens, metrics = model.generate(prompts_text, sampling_params)

    logs = ""
    for i, generated_text in enumerate(outputs):
        problems[ids[i]]["final_answer"] = {
            "response": generated_text,
            "prompt_tokens": tokens[i]["prompt"] if tokens[i] is not None else 0,
            "generated_tokens": tokens[i]["generated"] if tokens[i] is not None else 0,
            "total_tokens": tokens[i]["total"] if tokens[i] is not None else 0,
            "metrics": metrics[i]
        }
        if i % 100 == 0:
            logs += "############### Prompt for individual question {} ###############\n".format(
                p_id)
            logs += str(prompts_text[i]) + "\n"
            logs += "############### Response ###############\n"
            logs += generated_text + "\n\n"

    with open(os.path.join(output_path, "log.MD"), "a") as f:
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
    
    problems = generate_answers(model, test_questions, selected_examples, out_test_path, sampling_params)

    with open(os.path.join(out_test_path, "generations.json"), "w") as f:
        json.dump(problems, f, indent=4)
