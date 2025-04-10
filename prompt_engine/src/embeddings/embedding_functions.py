import torch
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

from angle_emb import AnglE
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class BaseEmbedding:
    """
        Base Embedding Function

        Args:
            embedding_path: path to the model
            emb_size: size of the embeddings
    """
    def __init__(self, embedding_path, emb_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_path = embedding_path
        self.emb_size = emb_size

    def embed(self, text):
        raise NotImplementedError

    def get_vectors(self, input, batch_size=32):
        """
            Encode the input texts using batches

            Args:
                input: input texts

            Returns:
                output: embeddings of the input texts
        """
        if isinstance(input, list):
            output = []
            for i in tqdm(range(0, len(input), batch_size), desc="Embedding questions..."):
                batch_input = input[i:i+batch_size]
                output.extend(self.embed(batch_input))
            return output
        else:
            return self.embed(input)


class UaeLargeEmbedding(BaseEmbedding):
    """
        UAE Large Embedding Function

        Args:
            embedding_path: path to the model
    """
    def __init__(self, embedding_path, *args, **kwargs):
        super().__init__(embedding_path, 1024, *args, **kwargs)
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            self.model = AnglE.from_pretrained(embedding_path, device="cuda:"+str(n_gpus-1), pooling_strategy='cls').cuda()
        else:
            self.model = AnglE.from_pretrained(embedding_path, pooling_strategy='cls').cuda()
        
        self.params = {
            "vectors_config": VectorParams(size=self.emb_size, distance=Distance.COSINE)
        }
        

    def embed(self, text):
        vec = self.model.encode(text, to_numpy=True)
        return vec.tolist()



class SentenceTransformersEmbedding(BaseEmbedding):
    """
        SentenceTransformer Embedding Function

        Args:
            embedding_path: path to the model
    """
    def __init__(self, embedding_path, *args, **kwargs):
        super().__init__(embedding_path, *args, **kwargs)
        model_kwargs = {
            "torch_dtype": "auto",
        }
        n_gpus = torch.cuda.device_count()
        
        if n_gpus > 1:
            self.model = SentenceTransformer(embedding_path, trust_remote_code=True, device="cuda:"+str(n_gpus-1), model_kwargs=model_kwargs)
        else:
            self.model = SentenceTransformer(embedding_path, trust_remote_code=True, model_kwargs=model_kwargs)
            
        self.emb_size = self.model.get_sentence_embedding_dimension()
        self.params = {
            "vectors_config": VectorParams(size=self.emb_size, distance=Distance.COSINE)
        }

    def embed(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    

class AutoModelEmbedding(BaseEmbedding):
    """
        AutoModel Embedding Function

        Args:
            embedding_path: path to the model
    """
    def __init__(self, embedding_path, *args, **kwargs):
        super().__init__(embedding_path, *args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_path)
        self.model = AutoModel.from_pretrained(embedding_path)
        self.emb_size = self.model.config.hidden_size
        self.params = {
            "vectors_config": VectorParams(size=self.emb_size, distance=Distance.COSINE)
        }


    def embed(self, text):
        with torch.no_grad():
            encoded = self.tokenizer(text, truncation=True, padding=True, max_length=self.model.config.max_position_embeddings, return_tensors='pt')            
            embeddings = self.model(**encoded).last_hidden_state[:, 0, :]
        return embeddings.detach().numpy().tolist()



class SpladeEmbedding:
    """
        SPLADE Embedding Function

        Args:
            embedding_path: path to the model
    """

    def __init__(self, embedding_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_path)
        self.model = AutoModelForMaskedLM.from_pretrained(embedding_path)
        self.emb_size = 768
        self.params = {
            "vectors_config": {},
            "sparse_vectors_config": {
                "text": SparseVectorParams(index=SparseIndexParams())
            }
        }
        self.idx2token = {idx: token for token, idx in self.tokenizer.get_vocab().items()}


    def embed(self, text):
        
        # Compute vector from logits and attention mask using ReLU, log, and max operations.
        with torch.no_grad():
            tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            output = self.model(**tokens)

        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vectors = max_val.squeeze()
        if vectors.dim() == 1:
            print("si")
            vectors = [vectors]

        embeddings = []
        for vec in vectors:
            # Extract indices and values of non-zero elements in the vector
            cols = vec.nonzero().squeeze().cpu().tolist()
            weights = vec[cols].cpu().tolist()
            sparse_dict = dict(zip(cols, weights))
            embeddings.append(sparse_dict)

        return embeddings



    def get_vectors(self, input, batch_size=64):
        """
            Encode the input texts using batches

            Args:
                input: input texts

            Returns:
                output: embeddings of the input texts
        """
        if isinstance(input, list):
            output = []
            for i in tqdm(range(0, len(input), batch_size), desc="Embedding questions..."):
                batch_input = input[i:i+batch_size]
                output.extend(self.embed(batch_input))
            return output
        else:
            return self.embed(input)   