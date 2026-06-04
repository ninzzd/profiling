from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import cProfile
import pstats
from io import StringIO
import time

MODEL_ID = "google/gemma-4-E2B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")

# Prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

# Process input
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Profiling wrapper around inference
profiler = cProfile.Profile()
profiler.enable()
# Generate output
outputs = model.generate(**inputs, max_new_tokens=1024)  # type: ignore
profiler.disable()
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)


# Parse inference output
parsed = processor.parse_response(response)
print('Inference output:')
print(f'{parsed["role"]}:{parsed["content"]}')
print()

# Profiling output and stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
stats.sort_stats('cumulative').dump_stats('gemma4_pyprof.prof')