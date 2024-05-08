from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from jinja2.exceptions import TemplateError

from .prompt_templates import jinja_chatml_template

from .logger import init_logger

logger = init_logger(__name__)

class Model():
    def __init__(self, vllm_params):
        print(vllm_params)
        self.model = LLM(**vllm_params)
        self.tokenizer = self.model.get_tokenizer()
        if self.tokenizer.default_chat_template is None:
            logger.info(f"No default template detected. Using ChatML as default...")
            self.tokenizer.chat_template = jinja_chatml_template

    def load_tokenizer(self, model):
        tokenizer = get_tokenizer(model, trust_remote_code=True)
        return tokenizer

    def get_chat_template(self, system_prompt, examples, question, add_system_prompt=True):
        chat_template = []
        if add_system_prompt:
            chat_template.append({"role": "system", "content": system_prompt})

        # Add few-shots examples
        for i, example in enumerate(examples):
            if i == 0 and not add_system_prompt:
                q = f"{system_prompt}\n{example['question']}"
            else:
                q = example["question"]

            chat_template.append({"role": "user", "content": example["question"]})
            chat_template.append({"role": "assistant", "content": example["answer"]})

        # Add the question to the template
        chat_template.append({"role": "user", "content": question})
        return chat_template

    def generate_prompt(self, system_prompt, examples, question, add_generation_prompt=True):
        try:
            return self.tokenizer.apply_chat_template(
                    self.get_chat_template(system_prompt, examples, question),
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    add_system_prompt=True
                ).strip()
        except TemplateError:
            try:
                return self.tokenizer.apply_chat_template(
                        self.get_chat_template(system_prompt, examples, question),
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        add_system_prompt=False
                    ).strip()
            except:
                self.tokenizer.chat_template = jinja_chatml_template
                return self.tokenizer.apply_chat_template(
                    self.get_chat_template(system_prompt, examples, question),
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    add_system_prompt=True
                ).strip()

    def generate(self, prompts, sampling_params):
        return self.model.generate(prompts, sampling_params)
