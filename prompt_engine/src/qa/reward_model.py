from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, pipeline
from src.utils.model import Model
# from prometheus_eval_ray.vllm import VLLM


class LLMEvaluator:
    def __init__(self, model_id):
        self.model = Model(model_id)
    

class ArmoRM:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, device_map=device_map, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length
        self.attributes = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence', 'helpsteer-complexity',
                           'helpsteer-verbosity','ultrafeedback-overall_score', 'ultrafeedback-instruction_following', 
                           'ultrafeedback-truthfulness', 'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe',
                           'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity', 'code-style',
                           'code-explanation','code-instruction-following','code-readability']

    def __call__(self, messages: List[Dict[str, str]]):
        """
        messages: chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            multi_obj_rewards = output.rewards.cpu().float()
            # The preference score for the response, aggregated from the 
            # multi-objective rewards with the gating layer
            preference_score = output.score.cpu().float()
        
        rewards = multi_obj_rewards[0, :19].tolist()
        
        return float(preference_score), rewards
    
    def get_str_rewards(self, messages: List[Dict[str, str]]):
        preference_score, multi_obj_rewards = self.__call__(messages)
        scores = {
            "ArmoRM_score": preference_score
        }

        for i in range(len(multi_obj_rewards)):
            scores[self.attributes[i]] = multi_obj_rewards[i]
            
        return scores
    