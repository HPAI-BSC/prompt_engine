import os
import time
import subprocess
import asyncio
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import requests
from requests.exceptions import RequestException

from .model import BaseModel
from .logger import init_logger

logger = init_logger(__name__)

def wait_server(base_url, max_wait_time=900, check_interval=10):
    def is_server_ready(base_url):
        try:
            response = requests.get(base_url)
            return response.status_code == 200
        except RequestException:
            return False
        
    logger.info(f"Waiting for VLLM server to start (up to {max_wait_time / 60} minutes)...")
    elapsed_time = 0
    while elapsed_time < max_wait_time:
        if is_server_ready(base_url):
            logger.info("VLLM server is ready!")
            break
        time.sleep(check_interval)
        elapsed_time += check_interval
        if elapsed_time % 60 == 0:
            logger.info(f"Elapsed time: {elapsed_time / 60} minutes")
    else:
        raise TimeoutError("VLLM server did not start within the expected time.")

class OpenAIModel(BaseModel):
    def __init__(self, vllm_params, ip="localhost:6378"):
        super().__init__(vllm_params)
        self.ip = ip
        self.model = vllm_params["model"]
        self.allowed_params = ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'max_tokens', 'stop', 'seed']
        self.allowed_extra = [
            "se_beam_search", "top_k", "min_p", "repetition_penalty", "length_penalty", "stop_token_ids", 
            "include_stop_str_in_output", "ignore_eos", "min_tokens", "skip_special_tokens", "spaces_between_special_tokens", 
            "truncate_prompt_tokens", "allowed_token_ids", "prompt_logprobs", "add_special_tokens", "response_format", 
            "guided_json", "guided_regex", "guided_choice", "guided_grammar", "guided_decoding_backend", 
            "guided_whitespace_pattern", "priority", "logits_processors"
        ]

        if "localhost" in ip and os.path.exists(vllm_params["model"]):
            self.start_vllm_server(vllm_params) # Start VLLM server locally
        else:
            if not self.tokenizer:
                # Using an external OpenAI compatible model. Set properly your environment variables
                logger.info(f"Using external VLLM server on {os.environ['API_URL']}")
                self.client = AsyncOpenAI(
                    api_key=os.environ["API_KEY"],
                    base_url=os.environ["API_URL"],
                )
            else:
                # Using an internal already created VLLM server. Useful when you have a VLLM server running on a different machine, 
                # or in a multinode setup. 
                logger.info(f"VLLM server already running on {ip}")
                self.client = OpenAI(
                    api_key="EMPTY",
                    base_url=f"http://{ip}/v1",
                    timeout=1500
                )


    def start_vllm_server(self, vllm_params):
        logger.info(f"Starting VLLM server...")
        command = [
            "vllm", "serve", vllm_params["model"],
            "--host", "127.0.0.1",
            "--port", "6378",
            "--dtype", str(vllm_params["dtype"]) if "dtype" in vllm_params else "bfloat16",
            "--tensor-parallel-size", str(vllm_params["tensor_parallel_size"]) if "tensor_parallel_size" in vllm_params else "1",
            "--max-model-len", str(vllm_params["max_model_len"]) if "max_model_len" in vllm_params else "8192",
            "--gpu-memory-utilization", str(vllm_params["gpu_memory_utilization"]) if "gpu_memory_utilization" in vllm_params else "0.9",
            "--seed", str(vllm_params["seed"]) if "seed" in vllm_params else "42",
        ]
        with open("vllm_output.log", "a") as log_file:
            self.process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)

        wait_server("http://127.0.0.1:6378/health")

        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://127.0.0.1:6378/v1",
            timeout=1500
        )

    def generate(self, prompts, sampling_params, batch_size=100, retry=0):
        for param in self.generation_config:
            if param not in sampling_params:
                sampling_params[param] = self.generation_config[param]
        params = {k: v for k, v in sampling_params.items() if k in self.allowed_params}
        extra_params = {k: v for k, v in sampling_params.items() if k in self.allowed_extra}

        if "temperature" in params:
            params["temperature"] += (0.05 * retry)
            params["temperature"] = min(1.0, params["temperature"])
        if "max_tokens" in params:
            params["max_tokens"] += (100 * retry)
        logger.info(f"Sampling parameters: {params}")
        logger.info(f"Extra parameters: {extra_params}")

        # If prompts is a list of lists, generate using chat completions
        if isinstance(prompts[0], list):
            return asyncio.run(self.generate_chat_parallel(prompts, params, extra_params, rpm_limit=batch_size))
        
        # If prompts is a list of strings, generate using text completions. Divide into batches to avoid timeout exceptons
        batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
        outputs = []
        tokens = []
        for batch in tqdm(batches, desc="Generating prompts"):
            completions = self.client.completions.create(
                model=self.model,
                prompt=batch,
                echo=False,
                n=1,
                stream=False,
                **params,
                extra_body=extra_params
            )
            outputs.extend([completion.text.strip() for completion in completions.choices])

            # Token counting returns the total tokens in all the prompts, not individual prompts.
            logger.info(f"Prompt tokens: {completions.usage.prompt_tokens}, Generated tokens: {completions.usage.completion_tokens}, Total tokens: {completions.usage.total_tokens}")
            tokens.extend([{
                "prompt": completions.usage.prompt_tokens / len(batch),
                "generated": completions.usage.completion_tokens / len(batch),
                "total": completions.usage.total_tokens / len(batch)
            }] * len(batch))

        assert len(outputs) == len(prompts), f"Number of outputs ({len(outputs)}) is different from the number of prompts ({len(prompts)})"
        assert len(tokens) == len(prompts), f"Number of tokens ({len(tokens)}) is different from the number of prompts ({len(prompts)}"
        metrics = [None] * len(prompts)
        return outputs, tokens, metrics 


    async def generate_chat_parallel(self, prompts, params, extra_params, rpm_limit=10):
        logger.info(f"Generating chat in parallel with an RPM limit of {rpm_limit}...")
        
        semaphore = asyncio.Semaphore(rpm_limit)  # Limit concurrent requests
        delay = 60.0  # Time interval between requests (in seconds)
        progress_bar = tqdm(total=len(prompts), desc="Processing Requests", unit="req")

        async def generate_single_chat(prompt, index):
            async with semaphore:
                for i in range(5):
                    await asyncio.sleep(delay)
                    try:
                        completion = await self.client.chat.completions.create(
                            model=self.model,
                            messages=prompt,
                            n=1,
                            stream=False,
                            **params,
                            extra_body=extra_params
                        )
                        response = completion.choices[0].message.content.strip()
                        usage = {
                            "prompt": completion.usage.prompt_tokens,
                            "generated": completion.usage.completion_tokens,
                            "total": completion.usage.total_tokens
                        }
                        progress_bar.update(1)
                        return response, usage
                    except Exception as e:
                        logger.info(f"[{index}] failed: {e}")  
                        logger.info(f"Retry {i+1} of 5")
            return None, None

        # Launch all requests in parallel
        logger.info(f"Launching {len(prompts)} requests...")
        tasks = [generate_single_chat(prompt, i) for i, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        outputs = [res[0] if isinstance(res[0], str) else "" for res in results]
        tokens = [res[1] if isinstance(res[1], dict) else None for res in results]
        metrics = [None] * len(outputs)
        progress_bar.close()
        logger.info("All requests completed.")
        return outputs, tokens, metrics

    
    def destroy(self):
        if "localhost" in self.ip:
            logger.info("Shutting down VLLM server...")
            self.process.terminate()
            self.process.wait()
        super().destroy()