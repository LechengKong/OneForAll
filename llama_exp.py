import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

## v2 models
model_path = "openlm-research/open_llama_3b_v2"

## v1 models
# model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "Q: Among machine learning, biology, and chemistry, which one is computer science most pertained to?\nA:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids.to("cuda:0"), max_new_tokens=32
)
print(tokenizer.decode(generation_output[0]))
print(torch.cuda.max_memory_allocated("cuda:0"))
