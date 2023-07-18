import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    model_id = "RWKV/rwkv-raven-1b5"
    model = AutoModelForCausalLM.from_pretrained(model_id).float()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    exportPath = sys.argv[1] if (sys.argv[1] is not None) else "rwkv-raven-1b5-fp32.flm";
    torch2flm.tofile(exportPath, model, tokenizer)
