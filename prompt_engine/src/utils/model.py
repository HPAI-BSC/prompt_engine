import os
from abc import ABC, abstractmethod
from dataclasses import asdict

from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams
import torch
import gc
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

from .prompt_templates import jinja_chatml_template

from .logger import init_logger

logger = init_logger(__name__)

class BaseModel(ABC):
    def __init__(self, vllm_params):
        self.model_name = os.path.basename(vllm_params["model"])
        self.tokenizer = self.load_tokenizer(vllm_params["model"])
        self.generation_config = self.load_generation_config(vllm_params["model"])
    
    def load_tokenizer(self, model):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            if tokenizer.chat_template is None:
                logger.info("No default template detected. Using ChatML as default...")
                tokenizer.chat_template = jinja_chatml_template
            else:
                logger.info("Chat template detected. Using it as default...")
            return tokenizer
        except:
            logger.warning(f"Failed to load tokenizer for model: {model}")
            return None

    def load_generation_config(self, model):
        try:
            config = GenerationConfig.from_pretrained(model).to_diff_dict()
            config.pop("transformers_version", None)
            logger.info(f"Model generation config: {config}")
            return config
        except:
            logger.warning("Failed to load generation config.")
            return {}
        
    def get_messages(self, system_prompt, examples, question, add_system_prompt=True):
        messages = [{"role": "system", "content": system_prompt}] if add_system_prompt else []

        for example in examples:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["answer"]})
        messages.append({"role": "user", "content": question})
        return messages
    
    def generate_prompt(self, system_prompt, examples, question, add_generation_prompt=True):
        messages = self.get_messages(system_prompt, examples, question)
        if not self.tokenizer:
            return messages

        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            ).strip()
        except Exception as e:
            logger.error(f"Failed to apply chat template: {e}")
            logger.info("Trying to use ChatML as default template...")
            self.tokenizer.chat_template = jinja_chatml_template
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            ).strip()
    
    def generate_prompt_single_turn(self, system_prompt, examples, question, add_generation_prompt=True):
        examples_text = "\n".join(
            [f"Example {i+1})\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n" for i, ex in enumerate(examples)]
        ) + f"\n\nThis is the question you have to answer:\nQuestion: {question}\n\n" if examples else question

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": examples_text}] if system_prompt else [{"role": "user", "content": examples_text}]

        if self.tokenizer:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            ).strip()
        return messages
    
    @abstractmethod
    def generate(self, prompts, sampling_params, batch_size=100, retry=0):
        pass

    def destroy(self):
        gc.collect()
        torch.cuda.empty_cache()

class Model(BaseModel):
    def __init__(self, vllm_params):
        super().__init__(vllm_params)
        self.allowed_params = [
            'temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'max_tokens', 'stop', 'seed', "top_k", "min_p", 
            "repetition_penalty", "length_penalty", "stop_token_ids", "include_stop_str_in_output", "ignore_eos", "min_tokens",
            "skip_special_tokens", "spaces_between_special_tokens", "truncate_prompt_tokens", "allowed_token_ids", "prompt_logprobs",
            "add_special_tokens"
        ]
        self.model = LLM(**vllm_params)

        
    def generate(self, prompts, sampling_params, batch_size=100, retry=0):
        for param in self.generation_config:
            if param not in sampling_params:
                sampling_params[param] = self.generation_config[param]

        sampling_params = {k: v for k, v in sampling_params.items() if k in self.allowed_params}
        if "temperature" in sampling_params:
            sampling_params["temperature"] += (0.05 * retry)
            sampling_params["temperature"] = min(1.0, sampling_params["temperature"])
        if "max_tokens" in sampling_params:
            sampling_params["max_tokens"] += (100 * retry)
        
        sampling_params = SamplingParams(**sampling_params)
        completions =  self.model.generate(prompts, sampling_params)

        outputs = [completion.outputs[0].text.strip() for completion in completions]
        tokens = [
            {
                "prompt": len(completion.prompt_token_ids),
                "generated": len(completion.outputs[0].token_ids),
                "total": len(completion.prompt_token_ids) + len(completion.outputs[0].token_ids)
            } 
            for completion in completions
        ]
        metrics = [asdict(completion.metrics) for completion in completions]
        
        return outputs, tokens, metrics 

    def destroy(self):
        super().destroy()
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.model.llm_engine.model_executor