---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
tags:
- abliterated
- uncensored
- huihui
- qwen3

---

# huihui-ai/Huihui-Qwen3.8-27B-abliterated


This is an uncensored version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) created with abliteration (see [remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers) to know more about it).
This is a crude, proof-of-concept implementation to remove refusals from an LLM model without using TransformerLens.

## Latest update

Only layers **18 to 51** have been ablated (Previously, The first 15 layers were retained without ablation), while the other layers remain unablated. 
This helps retain more of the original model’s performance. MTP and visual has not been modified.

If you want to download a previous version, just use the following command:

```
hf download huihui-ai/Huihui-Qwen3.8-27B-abliterated --local-dir ./huihui-ai/Huihui-Qwen3.8-27B-abliterated --revision d42ca89
```


## Note
The first 15 layers were retained without ablation. MTP and visual has not been modified.

## ollama

Please use the latest version of [ollama](https://github.com/ollama/ollama/releases)

You can use [huihui_ai/Qwen3.8-abliterated](https://ollama.com/huihui_ai/Qwen3.8-abliterated) directly, 
```
ollama run huihui_ai/Qwen3.8-abliterated
```


## Usage
You can use this model in your applications by loading it with Hugging Face's `transformers` library:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import os
import signal
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into huihui-ai/Huihui-Qwen3.8-27B-abliterated base model and save the full model."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="huihui-ai/Huihui-Qwen3.8-27B-abliterated",
        help="HuggingFace repo or local path of the base model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for loading the base model (default: bfloat16).",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading (e.g. 'cpu', 'auto').",
    )
    return parser.parse_args()

def main():
    cpu_count = os.cpu_count()
    print(f"Number of CPU cores in the system: {cpu_count}")
    half_cpu_count = cpu_count // 2
    os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
    os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)
    torch.set_num_threads(half_cpu_count)

    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"MKL threads: {os.getenv('MKL_NUM_THREADS')}")
    print(f"OMP threads: {os.getenv('OMP_NUM_THREADS')}")

    args = parse_args()

    # Load the model and tokenizer
    print(f"Load Model {args.base_model} ... ")

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    messages = []
    class CustomTextStreamer(TextStreamer):
        def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
            super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
            self.generated_text = ""
            self.stop_flag = False
            self.init_time = time.time()  # Record initialization time
            self.end_time = None  # To store end time
            self.first_token_time = None  # To store first token generation time
            self.think_tokens_count = 0  # To track total think tokens
            self.token_count = 0  # To track total tokens

        def on_finalized_text(self, text: str, stream_end: bool = False):
            if self.first_token_time is None and text.strip():  # Set first token time on first non-empty text
                self.first_token_time = time.time()
            if stream_end:
                self.end_time = time.time()  # Record end time when streaming ends

            self.generated_text += text
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            self.token_count += len(tokens)
            if self.think_tokens_count == 0 and "</think>" in self.generated_text:
                self.think_tokens_count = self.token_count
            print(text, end="", flush=True)

            if self.stop_flag:
                raise StopIteration

        def stop_generation(self):
            self.stop_flag = True
            self.end_time = time.time()  # Record end time when generation is stopped

        def get_metrics(self):
            """Returns initialization time, first token time, first token latency, end time, total time, total tokens, and tokens per second."""
            if self.end_time is None:
                self.end_time = time.time()  # Set end time if not already set
            total_time = self.end_time - self.init_time  # Total time from init to end
            tokens_per_second = self.token_count / total_time if total_time > 0 else 0
            first_token_latency = (self.first_token_time - self.init_time) if self.first_token_time is not None else None
            metrics = {
                "init_time": self.init_time,
                "first_token_time": self.first_token_time,
                "first_token_latency": first_token_latency,
                "end_time": self.end_time,
                "total_time": total_time,  # Total time in seconds
                "total_tokens": self.token_count,
                "think_tokens_count": self.think_tokens_count,
                "real_tokens_count": self.token_count - self.think_tokens_count,
                "tokens_per_second": tokens_per_second
            }
            return metrics

    def generate_stream(model, tokenizer, messages, enable_thinking, skip_prompt, skip_special_tokens, max_new_tokens):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(model.device)

        streamer = CustomTextStreamer(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)

        def signal_handler(sig, frame):
            streamer.stop_generation()
            print("\n[Generation stopped by user with Ctrl+C]")

        signal.signal(signal.SIGINT, signal_handler)

        print("Response: ", end="", flush=True)
        try:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer
            )
            del generated_ids
        except StopIteration:
            print("\n[Stopped by user]")

        del inputs
        torch.cuda.empty_cache()
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        return streamer.generated_text, streamer.stop_flag, streamer.get_metrics()

    skip_prompt=True
    skip_special_tokens=True
    enable_thinking=False
    
    while True:
        print(f"skip_prompt = {skip_prompt}.")
        print(f"skip_special_tokens = {skip_special_tokens}.")
        print(f"enable_thinking = {enable_thinking}.")

        user_input = input("User: ").strip()
        if user_input.lower() == "/exit":
            print("Exiting chat.")
            break
        if user_input.lower() == "/clear":
            messages = []
            print("Chat history cleared. Starting a new conversation.")
            continue
        if user_input.lower() == "/skip_prompt":
            skip_prompt = not skip_prompt
            continue
        if user_input.lower() == "/skip_special_tokens":
            skip_special_tokens = not skip_special_tokens
            continue
        if user_input.lower() == "/enable_thinking":
            enable_thinking = not enable_thinking
            continue
        if not user_input:
            print("Input cannot be empty. Please enter something.")
            continue

        messages.append({"role": "user", "content": user_input})
        response, stop_flag, metrics = generate_stream(model, tokenizer, messages, enable_thinking, skip_prompt, skip_special_tokens, 40960)
        print("\n\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        print("", flush=True)

        if stop_flag:
            continue
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

```
### Usage Warnings


 - **Risk of Sensitive or Controversial Outputs**: This model’s safety filtering has been significantly reduced, potentially generating sensitive, controversial, or inappropriate content. Users should exercise caution and rigorously review generated outputs.

 - **Not Suitable for All Audiences**: Due to limited content filtering, the model’s outputs may be inappropriate for public settings, underage users, or applications requiring high security.

 - **Legal and Ethical Responsibilities**: Users must ensure their usage complies with local laws and ethical standards. Generated content may carry legal or ethical risks, and users are solely responsible for any consequences.

 - **Research and Experimental Use**: It is recommended to use this model for research, testing, or controlled environments, avoiding direct use in production or public-facing commercial applications.

 - **Monitoring and Review Recommendations**: Users are strongly advised to monitor model outputs in real-time and conduct manual reviews when necessary to prevent the dissemination of inappropriate content.

 - **No Default Safety Guarantees**: Unlike standard models, this model has not undergone rigorous safety optimization. huihui.ai bears no responsibility for any consequences arising from its use.


### Donation
##### Your donation helps us continue our further development and improvement, a cup of coffee can do it.
- bitcoin:
```
  bc1qqnkhuchxw0zqjh2ku3lu4hq45hc6gy84uk70ge
```
- Support our work on [Ko-fi](https://ko-fi.com/huihuiai)!

