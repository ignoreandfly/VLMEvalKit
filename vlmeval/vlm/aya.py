import warnings
import torch
from .base import BaseModel


class AyaLM(BaseModel):
    """Text-only LLM wrapper for VLMEvalKit.

    Images are stripped from the input; only the text portions of each message
    are forwarded to the model.  This provides a blind-LLM baseline on vision
    benchmarks.
    """

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='CohereLabs/tiny-aya-global', **kwargs):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        self.model.eval()

        self.kwargs = dict(
            max_new_tokens=512,
            do_sample=False,
        )
        self.kwargs.update(kwargs)

    def generate_inner(self, message, dataset=None):
        # Concatenate all text parts; silently drop image parts
        text = ' '.join(
            msg['value'] for msg in message if msg['type'] == 'text'
        ).strip()

        messages = [{'role': 'user', 'content': text}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt',
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **self.kwargs)

        # Decode only the newly generated tokens
        generated = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
