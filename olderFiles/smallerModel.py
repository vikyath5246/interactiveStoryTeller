from transformers import AutoTokenizer
from llama_cpp import Llama
import os

# Download the model first if you haven't (≈600MB)
# You can use wget or download manually from Hugging Face:
# wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Initialize the GGUF model
model_path = "./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Load the tokenizer from original model for chat template
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Initialize LLM with GPU acceleration if available
llm = Llama(
    model_path=model_path,
    n_ctx=2048,  # Context window size
    n_threads=8,  # CPU threads
    n_gpu_layers=-1 if os.getenv('USE_GPU') == '1' else 0,  # Use all GPU layers if available
)

def format_prompt(history, new_input):
    system_msg = """<|system|>
    You are an interactive storyteller. Continue the story with 2-3 sentences, then offer 3 numbered choices.
    Current Context: {context}
    Format: [Story text...] [1. Option 1] [2. Option 2] [3. Option 3]
    </s>
    """.format(context=", ".join(history[-3:]) if history else "New story")

    messages = [
        {"role": "system", "content": system_msg},
        *[{"role": "user", "content": msg} for msg in history],
        {"role": "user", "content": new_input}
    ]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def run_story_cli():
    history = []
    print("Welcome to Interactive Storyteller! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nYour input (or choose 1-3): ")
        
        if user_input.lower() == 'exit':
            break
        
        full_prompt = format_prompt(history, user_input)
        
        # Generate response using GGUF model
        response = llm(
            prompt=full_prompt,
            max_tokens=300,
            temperature=0.8,
            repeat_penalty=1.1,
            stop=["</s>"]  # Stop generation at end-of-sequence token
        )
        
        generated_text = response['choices'][0]['text'].strip()
        
        # Processing remains the same
        story_text = ""
        options = []
        
        lines = generated_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(("[1.", "[2.", "[3.")):
                option_text = line[3:].strip(']').strip()
                options.append(option_text)
            else:
                story_text += line + "\n"
        
        story_text = story_text.strip()
        
        print("\nStory continues:")
        print(story_text)
        
        if options:
            print("\nPlease choose an option:")
            for idx, option in enumerate(options, 1):
                print(f"{idx}. {option}")
            
            while True:
                choice = input("Your choice (1-3): ")
                if choice in ['1', '2', '3']:
                    break
                print("Please enter a valid choice (1-3).")
            
            history.extend([
                user_input,
                story_text,
                f"User chose option {choice}: {options[int(choice)-1]}",
                options[int(choice)-1]
            ])
        
        if len(history) > 6:
            history = history[-6:]

if __name__ == "__main__":
    run_story_cli()