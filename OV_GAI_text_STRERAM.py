import sys
import warnings
warnings.filterwarnings(action='ignore')
import os
import openvino_genai
from datetime import datetime
from rich.console import Console  #https://rich.readthedocs.io/en/stable/console.html
console = Console(width=100)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_screen()
print("\033[93;1m")  #light yellow
model_path = "DeepSeek-R1-Distill-Qwen-1.5B-int8-ov"
device = "GPU"
start = datetime.now()
print(f"Loading model {model_path} to {device}...")
tokenizer = openvino_genai.Tokenizer(model_path)
pipe = openvino_genai.LLMPipeline(model_path,tokenizer=tokenizer,device= "GPU")
delta = datetime.now() -start
print(f"{model_path} loaded in {delta}")
history = []

print("\033[94;1m")  #light blue bold
intro = """
▒▒▒▓▒▓▒▒▓▓▒▓▒▓▓▓▓▓▓▓▓▓█▓█▓▓█▓▓█▓▓█▓█▓██▓███▓██▓███▓
▒▒▓▒▒░▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█▓▓█▓█▓▓▒▒█▓█▓█▓█▓█▓██▓█▓█
▒▓▒▒▓   ▒▓▒▓▓▓▓▓▓▓  ▓▓█▓█▓█▓▓█▓▓░ ▒▓█▓███▓███▓█▓▓██
▒▓▒▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓ ░░▒▓▓▓▒▒▒▓▓▓█░ ░█▓█▓█▓██▓██▓██▓█
▓▒▓▓▒░ ░▒  ░    ░▓   ░▓░░ ░░  ▒▓░ ░█▓████▓██▓███▓██
▓▓▓▓▓  ░▒░ ▓▓▓▓  ▓ ░▒▓▓  ░░░░  ▓░ ▒▓█▓█▓██▓██▓█▒██▓
▓▒▓▒▓░ ░▒  ▓▓▓▓ ░▓  ▒▓▓ ░▓▓▓▓▓▓▓░ ░███▓██▓█▓███████
▓▓▓▓▓  ░▓ ░▓▓▓▓░ ▓▒░  █▒      ▒▓░ ▒▓▓██▓██▓██▓█▓█▓█
▓▓▓▓▓▓▓▓▓▓▓█▓█▓▓▓█▓▓█▓█▓█▓█▓█▓█▓██▓██▓██▓██▓██▓▓███
▓▓▓▓▓▓▓▒ ░ ▓▓▓▓▓░  ▒▓▓█░    ▒▓█▓█▓░  ▒▓██▓███████▓█
▓▓▓█▓▒ ▓▓▓█▒▒▓░ █▓█▓░░▓▒░█▓█▓ ▓█ ░▓██▓ ▒▓▓▒█▓█▓▒███
▓█▓▓▓ ▒▓█▓▓▓█░ █▓█▓▓▓ █░ ░░░ ░▓▒        ███████▓█▓█
▓▓█▓▓░ ▓▓█▓█▓▓ ▓▓█▓█░ █▒░██▒░▓█▒ ▓██▓███▓█▓█▓██████
▓█▓█▓▓▓    ░▓█▓░    ▓█▓░░█▓█▓ ███     ▒▓█▓▒██▓▒░▓██
▓█▓█▓█▓███▓█▓██▓████▓███▓████▓█▓██████████▓████▓██▓
██▓██▓█▓████▓██▓█▓█▓██▓████▓█▓████████▓█▓███▓███▓██
▓██▓████▓█▓███▓██▓██▓███▓██████▒░█▓█░ ███░ ██▓░░▓██
█▓██▓░▒▓█ █ ▓█░   ▒▒  ░▓█▓ ▓▓█▓█▓██▓▓▓█▓█▓▓█▓█▓▓█▓█
██▓██░▒█▓ █ █▓█▓░█▓▒░░ █▓░▓░▓████▓████▓██▓█████▓███
███▓█▓ ░ █▓   ▓█ ██▒▓▓ ▓▒▒██ █▓░ █▓█  ██▓░ ▓█▓░ ▓▓█
▓██████▓█████████▓█████▓██▓████▓███▓██▓██▓██▓██████

"""
print(intro)

def chat(tokenizer,pipe, history,prompt):
    def streamer(subword):
        print(subword, end="", flush=True)
        sys.stdout.flush()
        # Return flag corresponds whether generation should be stopped.
        # False means continue generation.
        return False
    history.append({"role": "user", "content": prompt})
    # credit to me and https://huggingface.co/docs/transformers/llm_tutorial
    model_inputs = tokenizer.apply_chat_template(history,
                                            add_generation_prompt=True)
    answer = pipe.generate(model_inputs, max_new_tokens=900, streamer=streamer)
    history.append({"role": "assistant", "content": answer})
    return history, answer

while True:
    userinput = ""
    print("\033[1;30m")  #dark grey
    print("Enter your text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
    print("\033[91;1m")  #red
    lines = sys.stdin.readlines()
    for line in lines:
        userinput += line + "\n"
    if "quit!" in lines[0].lower():
        print("\033[0mBYE BYE!")
        break
    print("\033[92;1m")
    history, new_message = chat(tokenizer,pipe,history,userinput)
    #console.print(new_message)  #already done by hthe streamer, but no fixed width