<div align="center" id="promptenginetop">
<img src="https://raw.githubusercontent.com/HPAI-BSC/prompt_engine/main/images/prompt_engine_logo.png" alt="logo" width="350" margin="10px"></img>

![GitHub License](https://img.shields.io/github/license/HPAI-BSC/prompt_engine)
![GitHub followers](https://img.shields.io/github/followers/HPAI-BSC)
![GitHub Repo stars](https://img.shields.io/github/stars/HPAI-BSC/prompt_engine)
![GitHub forks](https://img.shields.io/github/forks/HPAI-BSC/prompt_engine)
![GitHub watchers](https://img.shields.io/github/watchers/HPAI-BSC/prompt_engine)

</div>

<h2 align="center">
prompt_engine: Evaluate your model using advanced prompt strategies
</h2>

<p align="center">
| <a href="https://arxiv.org/abs/2409.15127"><b>Paper</b></a> | <a href="https://huggingface.co/collections/HPAI-BSC/healthcare-llms-aloe-family-6701b6a777f7e874a2123363"><b>Aloe Family Models</b></a> | <a href="https://hpai.bsc.es/"><b>HPAI Website</b></a> |
</p>

*Latest News* 🔥
- [2025/03] [**Cost-Effective, High-Performance Open-Source LLMs via Optimized Context Retrieval**](https://arxiv.org/abs/2409.15127) pre-print is now available in Arxiv! Main contributions:
  - Practical guide for cost-effective optimized context retrieval.
  - Improved the Pareto Frontier on MedQA with **open-source** models: DeepSeek-R1 and Aloe-Beta-70B.
  - We introduce **OpenMedQA**. a novel benchmark derived from MedQA, to rigorously evaluate open-ended medical question answering.
- [2024/09] [**Aloe-Beta**](https://huggingface.co/collections/HPAI-BSC/healthcare-llms-aloe-family-6701b6a777f7e874a2123363) is out! New medical SOTA models available in Hugginface!
- [2024/05] [**Aloe: A Family of Fine-tuned Open Healthcare LLMs**](https://arxiv.org/abs/2405.01886) is now available in Arxiv!
- [2024/04] [**Aloe-Alpha-8B**](https://huggingface.co/HPAI-BSC/Llama3-Aloe-8B-Alpha) is now available in Hugginface!



## About

This repository provides a comprehensive framework for evaluating large language models (LLMs) using various prompt engineering techniques to improve performance on medical benchmarks. Our main goal is to explore how prompt engineering affects the accuracy, reliability, and overall usefulness of LLMs in addressing complex medical scenarios. This repository was initially created to support the [Aloe](https://huggingface.co/HPAI-BSC/Llama3-Aloe-8B-Alpha) model.

Our research focuses on leveraging the reasoning capabilities of LLMs through advanced prompt engineering techniques for medical applications. Specifically, this repository enables the evaluation of models on multiple-choice question-answering (MCQA) benchmarks using:

- **Self-Consistency Chain-of-Thought (SC-CoT)**
- **Medprompt**: A technique proposed by Microsoft, adapted for open-source models. Our version maintains all original functionalities while adding extra customizations:
  - Support for custom embedding models
  - Compatibility with pre-created custom databases
  - Option to include a reranker model


Additionally, the repository supports Open-Ended QA datasets, allowing answer generation and evaluation using an automated LLM-as-a-judge approach. We have also included the novel **OpenMedQA** dataset.


## Implementation

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/HPAI-BSC/prompt_engine/main/images/prompt_engine_features.png">
    <img alt="prompt_engine_features" src="https://raw.githubusercontent.com/HPAI-BSC/prompt_engine/main/images/prompt_engine_logo.png" width=55%>
  </picture>
</p>

To effectively implement these novel techniques, we developed a specialized framework centered around accelerated inference speed and efficient data storage & retrieval mechanisms. Specifically, the architecture employs:
- [**VLLM**](https://github.com/vllm-project/vllm): Fast Inference Very Large Language Model library to facilitate rapid generation of responses in an efficient way. The repository allows:
  - The use of the Offline mode of vLLM.
  - The use of the vLLM OpenAI compatible server, for both, serving it locally or using an external API.
- Vector database: Vector database solution to facilitate the storage and computation of vector similarities required for setting up the Medprompt technique. Both frameworks work in the self-hosted mode, storing the database locally under the "databases" path. We integrated two different vector database solutions:
    - [**ChromaDB**](https://github.com/chroma-core/chroma):. Open-source embedding database, focused on the simplicity and efficiency. only dense vectors are allowed.
    - [**Qdrant**](https://qdrant.tech/): Embedding database focused on production-ready service with a convenient API. It allows to use the database client by creating a docker image, using the local memory or in the cloud. In our implementation we use the local memory. It allows Sparse Vectors. We recommend to use this database only when dealing with Sparse Vectors.


Indeed, central to our approach was providing flexibility and adaptability in executing diverse tasks related to evaluating LLMs on medical benchmarks. Users can easily configure a wide array of parameters according to their unique experimental designs or preferences by modifying simple YAML configuration files found within the designated config directory. 

## Usage guide

### Installation
A requirements file with the necessary packages is provided to install and execute this repo.

```
pip install -r requirements.txt
```

### Execution

To execute the test. First, make sure you configured properly a YAML configuration file. Then, execute the following script:

```
python prompt_engine/run.py configs/aloe_medqa.yaml
```

If you want to run a model already serving in your machine you can set the "--ip" and "--port" parameters:
```
python prompt_engine/run.py configs/qwen_72b.yaml --ip 127.0.0.1 --port 888
```

If the responses are already generated, the evaluation step can be ran by:
```
python prompt_engine/run_evaluation.py configs/qwen_72b.yaml
```

This is specially useful when evaluating **OpenMedQA** with a large LLM as a judge.

### Configure an execution
To configure an execution, a configuration file must be created. Some examples are included in the "/configs" folder. The configuration files define the parameters of the execution, model configuration, sampling parameters, and the evaluators (only OpenMedQA). The system will automatically identify which type of execution (SC-CoT, Medprompt or Open-Ended QA) is being asked and will run it. More details about the configuration parameters can be found [here](configs).

### Datasets
We include the following medical datasets formatted in the required format:

- MedQA
- MedMCQA
- PubMedQA
- MMLU (medical subsets)
- CareQA
- OpenMedQA

Other datasets can be used if they are in the required format. More details can be found in [Datasets](medprompt/datasets) section.


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
}
```


When the generations of an execution, the evaluation procedure starts, and generates the results storing 3 extra files:

- evaluation_TIMESTAMP.json: Stores the evaluation results.
    - Number of examples evaluated and the **accuracy**
    - Number of answers that the model has not been able to answer correctly (_no_pased_answers_)
    - Number of draws in the majority voting (and how many resulted or not in a correct prediction) (_draws_)
    - Number of times the model predicted each option (ABCD...) (_options_)
    - Statistics regarding the number of generated tokens, time spend inferencing the model, lengths of the prompts... (_statistics_)
    - The parameters of the execution (_config_)
    - Timestamp of the evaluation execution (_date_) and the last modification timestamp of the generations.json file (_generation_date_)
- times_TIMESTAMP.json: Saves information about the time spent in each stage of the process.
- incorrect_questions_TIMESTAMP: Stores the questions that resulted in an incorrect result after the majority voting.

The outputs for **OpenMedQA** executions are slightly different, as it is not a mutliple-choice benchmark, but it follows the main structure.

### Databases
When running Medprompt, we allow the utilization of custom databases. When a database is specified in the configuration, the generation of training/validation CoT examples will be skipped and the examples of the databases will be used instead. However, the database must have the same format as the output generations file, detailed above. Feel free to generate your custom databases to test the Medprompt technique. We recommend the utilization of our databases. Download the json files from [here](https://huggingface.co/collections/HPAI-BSC/medical-context-retrieval-rag-67b0e0b0589983db691217cd), and move the files to the "prompt_engine/databases" path.

### Citations
If you use this repository in a published work, please cite the following papers as sources:

Bayarri-Planas, J., Gururajan, A. K., and Garcia-Gasulla, D., 2024. Boosting Healthcare LLMs Through Retrieved Context. arXiv preprint arXiv:2409.15127.
```
@misc{bayarriplanas2025costeffectivehighperformanceopensourcellms,
      title={Cost-Effective, High-Performance Open-Source LLMs via Optimized Context Retrieval}, 
      author={Jordi Bayarri-Planas and Ashwin Kumar Gururajan and Dario Garcia-Gasulla},
      year={2025},
      eprint={2409.15127},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2409.15127}, 
}
```

Gururajan, A.K., Lopez-Cuena, E., Bayarri-Planas, J., Tormos, A., Hinjos, D., Bernabeu-Perez, P., Arias-Duart, A., Martin-Torres, P.A., Urcelay-Ganzabal, L., Gonzalez-Mallo, M. and Alvarez-Napagao, S., 2024. Aloe: A Family of Fine-tuned Open Healthcare LLMs. arXiv preprint arXiv:2405.01886.
```
@misc{gururajan2024aloefamilyfinetunedopen,
      title={Aloe: A Family of Fine-tuned Open Healthcare LLMs}, 
      author={Ashwin Kumar Gururajan and Enrique Lopez-Cuena and Jordi Bayarri-Planas and Adrian Tormos and Daniel Hinjos and Pablo Bernabeu-Perez and Anna Arias-Duart and Pablo Agustin Martin-Torres and Lucia Urcelay-Ganzabal and Marta Gonzalez-Mallo and Sergio Alvarez-Napagao and Eduard Ayguadé-Parra and Ulises Cortés Dario Garcia-Gasulla},
      year={2024},
      eprint={2405.01886},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.01886}, 
}
```








