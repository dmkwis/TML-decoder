from transformers import pipeline
import random
generator = pipeline('text-generation', model='gpt2')
current_sentence = ""
num_gens = 10
step_size = 3
cur_tokens = 0
num_steps = 4


for i in range(num_steps):
    cur_tokens += step_size
    result = generator(current_sentence, max_length=cur_tokens, num_return_sequences=num_gens)
    current_sentence = result[random.randint(0, num_gens - 1)]['generated_text']
    print(current_sentence)
