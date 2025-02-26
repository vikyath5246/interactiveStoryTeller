from flask import Flask, request, session, render_template, redirect
from transformers import AutoTokenizer
from llama_cpp import Llama
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Initialize model and tokenizer once
model_path = "./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=-1 if os.getenv('USE_GPU') == '1' else 0,
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

@app.route('/', methods=['GET'])
def home():
    # Initialize full_story in session if it doesn't exist
    if 'full_story' not in session:
        session['full_story'] = []
    return render_template('index.html', 
                         full_story=session['full_story'],
                         options=session.get('current_options', []))

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.form.get('user_input', '').strip()
    history = session.get('history', [])
    full_story = session.get('full_story', [])

    # Handle option selection
    current_options = session.get('current_options', [])
    if current_options and user_input in ['1', '2', '3']:
        choice_idx = int(user_input) - 1
        if 0 <= choice_idx < len(current_options):
            # Add the chosen option text to story history
            selected_option = f"You chose: {current_options[choice_idx]}"
            full_story.append(selected_option)
            session['full_story'] = full_story
            
            # Update history for model context
            history.extend([
                user_input,
                session['current_story'],
                selected_option,
                current_options[choice_idx]
            ])
            history = history[-6:]  # Keep last 3 interactions
            session['history'] = history
            session.pop('current_options', None)

    # Generate new content
    full_prompt = format_prompt(history, user_input)
    response = llm(
        prompt=full_prompt,
        max_tokens=300,
        temperature=0.8,
        repeat_penalty=1.1,
        stop=["</s>"]
    )
    
    generated_text = response['choices'][0]['text'].strip()
    
    # Parse response
    story_text = ""
    options = []
    for line in generated_text.split('\n'):
        line = line.strip()
        if line.startswith(("[1.", "[2.", "[3.")):
            options.append(line[3:].strip(']').strip())
        else:
            story_text += line + "\n"
    
    # Update story history
    full_story.append(story_text.strip())
    session['full_story'] = full_story
    session['current_story'] = story_text.strip()
    
    if options:
        session['current_options'] = options
    else:
        session.pop('current_options', None)
    
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)