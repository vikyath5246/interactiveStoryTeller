from transformers import pipeline, AutoTokenizer
import torch

# Set device to MPS if available, else use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model with 4-bit quantization for better performance
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
story_gen = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically uses GPU if available
    tokenizer=tokenizer
)
# print(next(story_gen.model.parameters()).device)

def format_prompt(history, new_input):
    # Create system message with context instructions
    system_msg = """<|system|>
    You are an interactive storyteller. Continue the story with 2-3 sentences, then offer 3 numbered choices.
    Current Context: {context}
    Format: [Story text...] [1. Option 1] [2. Option 2] [3. Option 3]
    </s>
    """.format(context=", ".join(history[-3:]) if history else "New story")

    # Build conversation history
    messages = [
        {"role": "system", "content": system_msg},
        *[{"role": "user", "content": msg} for msg in history],
        {"role": "user", "content": new_input}
    ]
    
    # Apply TinyLlama's chat template
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def run_story_cli():
    history = []
    print("Welcome to Interactive Storyteller! (Type 'exit' to quit)")
    
    while True:
        # Get user input
        user_input = input("\nYour input (or choose 1-3): ")
        
        if user_input.lower() == 'exit':
            break
        
        # Generate story continuation
        full_prompt = format_prompt(history, user_input)
        response = story_gen(
            full_prompt,
            max_new_tokens=300,
            temperature=0.8,
            repetition_penalty=1.1,
            do_sample=True
        )[0]['generated_text']
        
        # Initialize variables to store story and options
        story_text = ""
        options = []
        
        # Split the response into lines
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("[1.") or line.startswith("[2.") or line.startswith("[3."):
                # Extract option text
                option_text = line[3:].strip(']').strip()
                options.append(option_text)
            else:
                # Accumulate story text
                story_text += line + "\n"
        
        # Remove any trailing newline characters
        story_text = story_text.strip()
        
        # Display the story continuation
        print("\nStory continues:")
        print(story_text)
        
        # Display the options
        if options:
            print("\nPlease choose an option:")
            for idx, option in enumerate(options, 1):
                print(f"{idx}. {option}")
            
            # Prompt the user to make a choice
            while True:
                choice = input("Your choice (1-3): ")
                if choice in ['1', '2', '3']:
                    break
                else:
                    print("Please enter a valid choice (1-3).")
            
            # Append the user's choice to history
            history.append(user_input)
            history.append(story_text)
            history.append(f"User chose option {choice}: {options[int(choice)-1]}")
            
            # Append the chosen option to history
            history.append(options[int(choice)-1])
        else:
            print("\nNo options available.")
            history.append(user_input)
            history.append(story_text)
        
        # Keep history manageable (last 3 exchanges)
        if len(history) > 6:
            history = history[-6:]

if __name__ == "__main__":
    run_story_cli()