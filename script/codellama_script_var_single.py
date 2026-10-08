import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, CodeLlamaTokenizer, LogitsProcessorList, BitsAndBytesConfig, LogitsProcessor

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

import json


code_type = 'decompile'
model_type = 'CodeLlama-34b-Instruct'

class SafeLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
        return scores

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def modify(orig):
    new = orig.split('\n')
    idx = 0
    while True:
        if 'MASK' not in new[idx]:
            idx += 1
        else:
            break

        if idx == len(new):
            break

    if idx < len(new):
        output = '\n'.join(new[idx:])
    else:
        output = orig
    return output


def main(
    load_8bit: bool = False,
    base_model: str = "codellama/CodeLlama-34b-Instruct-hf",
    lora_weights: str = "var",
    prompt_template: str = "codellama",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    arch: str = "",
    opt: str= "",
):
    lora_weights = f"{lora_weights}/{arch}/{opt}"

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = CodeLlamaTokenizer.from_pretrained(base_model, add_prefix_space=True)

    # ── Quantization config ──────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Determine GPU count and loading strategy ─────────────────────
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}", flush=True)

    # Single GPU - force everything on GPU 0
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        cache_dir='models/',
        local_files_only=True,
        trust_remote_code=True,
    )


    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        is_trainable=False,
    )


    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    safe_logits_processor = LogitsProcessorList([SafeLogitsProcessor()])

    def get_tokens_as_list(word_list):
        "Converts a sequence of words into a list of tokens"
        return tokenizer(word_list, add_special_tokens=False).input_ids


    def evaluate(
        instructionList,
        inputList=None,
        temperature=0.2,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=64,
        stream_output=False,
        **kwargs,
    ):

        prompt = prompter.generate_prompt(instructionList, inputList)
        inputs = tokenizer(prompt, return_tensors="pt")

        input_len = inputs["input_ids"].shape[1]

        # ── Truncate if too long to prevent OOM ──────────────────────
        if input_len > 2048:
            print(f"  WARNING: Truncating {input_len} -> 1024 tokens", flush=True)
            return "None"

        input_ids = inputs["input_ids"].to("cuda:0")
        attention_mask = inputs["attention_mask"].to("cuda:0")

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                do_sample=False,
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                remove_invalid_values=True,
                logits_processor=safe_logits_processor,
            )
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            return prompter.get_response(output) # llama2



    fileinput = f"dataset/{arch}/{opt}/test_set.json"
    fp = open(fileinput, 'r')

    fout =  open(f"var_output/{arch}_{opt}_var_infer.jsonl", 'w')
    for i, line in enumerate(fp.readlines()):
        output = {}
        output = json.loads(line)
        print (i)
        instruction = output['resym_instruction']
        inputs = modify(output["stripped"])

        res = evaluate(instruction, inputs)

        final = {}
        final.update(output)
        final["final"] = res
        json.dump(final, fout)

        fout.write('\n')
        fout.flush()


if __name__ == "__main__":
    fire.Fire(main)
