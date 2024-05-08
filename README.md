# prompt_engine: Evaluate your model using advanced prompt strategies

This repository serves as a comprehensive platform for evaluating large language models (LLMs) utilizing diverse prompt engineering techniques aimed at enhancing performance on medical benchmarks. Our goal is to explore how prompt engineering impact LLMs' accuracy, reliability, and overall usefulness in addressing complex medical scenarios. This repo was first created to support the [Aloe](https://huggingface.co/HPAI-BSC/Llama3-Aloe-8B-Alpha) model.

Central to our investigation were efforts to exploit the inherent reasoning capabilities of LLMs by employing sophisticated prompt engineering approaches towards medical applications. Among the techniques adopted include:
- **Self-Consistency Chain-of-Thought (SC-CoT)**: An iterative process wherein the LLM generates plausible explanations supporting each proposed solution before settling on a final answer. By encouraging systematic thinking, SC-CoT helps enhance both the confidence and veracity of generated responses.

- [**Medprompt**](https://github.com/microsoft/promptbase): A technique proposed by Microsoft, extending the traditional SC-COT. Medprompt introduces additional features specifically targeted at refining LLM behavior in medical settings. One key enhancement involves randomly shuffling provided choices before soliciting the LLM's response, thereby discouraging biases arising from predictable option orderings. Furthermore, integrating relevant case studies or "K nearest neighbor" (Knn) few-shot examples directly into prompts allows LLMs to learn from analogous situations and draw parallels between them, ultimately fostering better-rounded judgements.


## Implementation

To effectively implement these novel techniques, we developed a specialized framework centered around accelerated inference speed and efficient data storage & retrieval mechanisms. Specifically, the architecture employs:
- [**VLLM**](https://github.com/vllm-project/vllm): Fast Inference Very Large Language Model ibrary to facilitate rapid generation of responses in a efficient way.
- Vector database:  Vector database solution to facilitate the storage and computation of vector similarities required for setting up the Medprompt technique. Both frameworks work in the self-hosted mode, storing the database locally under the "databases" path. We integrated two different vector database solutions:
    - [**ChromaDB**](https://github.com/chroma-core/chroma):. Open-source embedding database, focused on the simplicity and efficiency. only dense vectors are allowed.
    - [**Qdrant**](https://qdrant.tech/): Embedding database focused on production-ready service with a convenient API. It allows to use the database client by creating a docker image, using the local memory or in the cloud. In our implementation we use the local memory. It allows Sparse Vectors. We recommend to use this database only when dealing with Sparse Vectors.


Indeed, central to our approach was providing flexibility and adaptability in executing diverse tasks related to evaluating LLMs on medical benchmarks. Users can easily configure a wide array of parameters according to their unique experimental designs or preferences by modifying simple YAML configuration files found within the designated config directory. 

## Usage guide

### Installation
To install and execute this repo, a requirements file is provided with the necessary packages.

```
pip install -r requirements.txt
```

To execute test. First make sure you configured properly a YAML configuration file. Then, execute the following script:

```
python prompt_engine/run.py configs/your_config.YAML
```

### Configure a execution
To configure a execution of Medprompt or Self-Consistency CoT ensembling, a configurtion file must be created. Some examples are included in the "/configs" folder. The configuration files defines the parameters of the execution, model configuration and sampling parameters. More details about the configuration parameters can be found [here](configs).

### Datasets
We include the following medical datasets formatted in the required format:

- MedQA
- MedMCQA
- PubMedQA
- MMLU (medical subsets)
- CareQA

Other datasets can be used if they are in the required format. More details in [Datasets](medprompt/datasets) section.


### Output
Each execution is stored in a specific path that is determined by the parameters selected and under the "outputs/" directory. The subdirectories generated will have this format:

```
    - outputs  
        -- MODEL_NAME
            -- embedding1
                --- DATASET_NAME
                    ---- SUBJECT (if dataset have subjects)
                        ----- N_ENSEMBLES
                            ----- Kk (5k, 4k...) + _database and/or _reranker if they exist in confg
            -- embedding2
                ...
                    ...
            -- SC-COT
                --- DATASET_NAME
                    ---- SUBJECT (if dataset have subjects)
                        ----- N_ENSEMBLES
                            ----- Kk (5k, 4k...)
            -- val
                --- DATASET_NAME
                    ---- SUBJECT (if dataset have subjects)
                        ----- N_ENSEMBLES
                            ------ Kk (5k, 4k...)
```

In the "val" directory are saved the generations of Medprompt of the training/validation sets, that are used for selecting the knn similar examples for the test questions. It is only generated when running Medprompt without database.

Each final directory will store the generations file of the exection, named "generations.json". The generations file holds the information of the question and the corresponding generations. The format of the generations is the following:

```
{
  "ID": {
    "question": "QUESTION",
    "correct_answer": "A",
    "options": {
      "A": "1",
      "B": "2",
      "C": "3",
      "D": "4"
    },
    "generations": [
      {
        "description": "DESCRITION OF THE QUESTION = QUESTION + SHUFFLED OPTIONS",
        "response": "RESPONSE OF THE MODEL",
        "original_answer": "A", # Original option selection of the model response
        "result": "C",  # Option selection in the original options order
        "order": "BDAC",    # Order of the options in the description
        "retries": 1,   # Number of retries until the model returned a valid answer
        "prompt_tokens": 684,   # Number of tokens in the prompt
        "generated_tokens": 95, # Number of generated tokens
        "metrics": {    # Time information of the inference returned by VLLM
          "arrival_time": 6843.604095897,
          "last_token_time": 6843.604095897,
          "first_scheduled_time": 1713348678.0800078,
          "first_token_time": 1713348678.198892,
          "time_in_queue": 1713341834.4759119,
          "finished_time": 1713348681.9018264
        }
      },
      {
          ... (N_ENSEMBLES TIMES)
      }
  },
  {
      "ID2": {
          ...
      }
  }

```


When the generations of a execution, the evaluation procedure starts, and generates the results storing 3 extra files:

- evaluation_TIMESTAMP.json: Stores the evaluation results.
    - Number of examples evaluated and the **accuracy**
    - Number of answers that the model has not been able of answering correctly (_no_pased_answers_)
    - Number of draws in the majority voting (and how many resulted or not in a correct prediction) (_draws_)
    - Number of times the model predicted each option (ABCD...) (_options_)
    - Statistics regarding the number of generated tokens, time spend inferencing the model, lengths of the prompts... (_statistics_)
    - The parameters of the execution (_config_)
    - Timestamp of the evalution execution (_date_) and the last modification timestamp of the generations.json file (_generation_date_)
- times_TIMESTAMP.json: Saves information about the time spend in each stage of the proces.
- incorrect_questions_TIMESTAMP: Stores the questions that resulted in a incorrect result after the majority voting.

### Databases
When running Medprompt, we allow the utilization of custom databases. When a databases is specified in the configuration, the generation of training/validation CoT examples will be skipped and the examples of the databases will be used instead. However, the database must have the same format as the output generations file, detailed above. Feel free to generate your custom databases to test the Medprompt technique.

### Citations
If you use this repository in a published work, please cite the following papers as source:

Gururajan, A.K., Lopez-Cuena, E., Bayarri-Planas, J., Tormos, A., Hinjos, D., Bernabeu-Perez, P., Arias-Duart, A., Martin-Torres, P.A., Urcelay-Ganzabal, L., Gonzalez-Mallo, M. and Alvarez-Napagao, S., 2024. Aloe: A Family of Fine-tuned Open Healthcare LLMs. arXiv preprint arXiv:2405.01886.







