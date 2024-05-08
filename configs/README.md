# Configuration files

The configuration file must be a YAML with the following entries:


    - vllm
        Defines the parameters of the model to use. See the oficial [VLLM repo](https://github.com/vllm-project/vllm/blob/05434764cd99990035779cf9a4ed86623b528825/vllm/entrypoints/llm.py) to see the full list of arguments. Mandatory parameters:
        - model (str)
            Path of the model to use. It can be a either a local or hugginface path.

    - sampling_params (optional)
         Sampling parameters for text generation. Take a look at the [official documentation](https://github.com/vllm-project/vllm/blob/05434764cd99990035779cf9a4ed86623b528825/vllm/sampling_params.py) to see the details and available parameters. Includint this parameters is optional. If not, default values will be used.

    - config:
        - dataset (str)
            Name of the dataset to test. See available datasets in medprompt/datasets.
        - subject (str)
            If dataset have different subjects, like MMLU, it allows to select which subject to test. If "all", all the subjected are executed. To launch multiple subjects, split them using commas: "anatomy,virology".
        - type (str ["medprompt", "sc-cot"])
            IMPORTANT. Type of execution to launch. It must be "medprompt" or "sc-cot".
        - vector_database (str ["chromadb", "qdrant"])
            DEFAULT is chromadb, use qdrant when using sparse vectors.
        - embedding (str)
            Path of the embedding model to use. This parameter is ignored if "sc-cot" type selected, and mandatory if "medprompt" is selected.
            Supported embedding models: 
                [UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)
                [BiomedNLP-BiomedBERT-base-uncased-abstract](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)
                [MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder)
                All [SentenceTransformers](https://sbert.net/) compatible models. WARNING. We only tested [SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral) and [pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings).
        - k (int)
            Number of few-shots to include in the prompt. Defaults to 5 with Medprompt and 3 with SC-COT
        - ensembles: (int. Default 5)
            Number of ensembles to run
        - overwrite (boolean. Default True)
            This parameter is ignored if "sc-cot" type selected. If selected, generates validation examples even if they already exist.
        - shuffle: (boolean)
            If selected, choices of the question are shuffled in each ensemble. Defaults to True if "medprompt", else False.
        - database (str. Optional)
            Optional. It allows to use a custom database of examples instead of generating the valdiation examples of the dataset. It requires the name of the database file formatted in the medprompt/datasets/databases. It must be a json file. If "sc-cot" this parameter will be ignored.
        - reranker (dict. Optional)
            - path (str)
                Path of the rearanker model to use. It can be local or hugginface model. Currently the only supported reranker model is [MedCPT-Cross-Encoder](https://huggingface.co/ncbi/MedCPT-Cross-Encoder)
            - n_rank (int)
                Optional. Defaults to 3. Indicates the number of KNN selected examples before reranking. The number of selected examples will be (n_rank * K)
