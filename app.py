import os
from flask import Flask, render_template, request, jsonify
from flask import send_from_directory
import asyncio
import time
from ai_orchestrator import ClaudeModel, GPTModel, GeminiModel, AIOrchestrator

app = Flask(__name__)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Configure available AI models
available_models = {
    "claude-sonnet": ClaudeModel("Claude 3 Sonnet"),
    "claude-opus": ClaudeModel("Claude 3 Opus", "claude-3-opus-20240229"),
    "gpt-4": GPTModel("GPT-4"),
    "gpt-3.5": GPTModel("GPT-3.5 Turbo", "gpt-3.5-turbo"),
    "gemini-pro": GeminiModel("Gemini 1.5 Pro"),
    "gemini-flash": GeminiModel("Gemini 1.5 Flash", "gemini-1.5-flash")
}

# Use Claude Opus as the orchestrator by default
default_orchestrator = available_models["claude-opus"]

@app.route('/')
def index():
    return render_template('index.html', available_models=list(available_models.keys()))

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    prompt = data.get('prompt', '')
    selected_models = data.get('models', [])
    orchestrator_model = data.get('orchestrator', 'claude-opus')
    
    # Validate input
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    if not selected_models:
        return jsonify({"error": "At least one model must be selected"}), 400
    
    # Create list of selected model instances
    models = [available_models[model] for model in selected_models if model in available_models]
    
    # Set up the orchestrator
    orchestrator_instance = available_models.get(orchestrator_model, default_orchestrator)
    orchestrator = AIOrchestrator(models, orchestrator_instance)
    
    # Process the request asynchronously
    start_time = time.time()
    
    # Create an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Gather responses from models
    responses = loop.run_until_complete(orchestrator.gather_responses(prompt))
    
    # Log raw responses for debugging
    print(f"Raw responses: {responses}")
    
    # Compile responses
    final_response = loop.run_until_complete(orchestrator.compile_responses(prompt, responses))
    loop.close()
    
    elapsed_time = time.time() - start_time
    
    # Format individual model responses for the UI
    model_responses = []
    for resp in responses:
        if resp["success"]:
            model_responses.append({
                "model": resp["model"],
                "response": resp["response"]
            })
        else:
            model_responses.append({
                "model": resp["model"],
                "response": f"Error: {resp.get('error', 'Unknown error')}"
            })
    
    return jsonify({
        "compiled_response": final_response,
        "individual_responses": model_responses,
        "processing_time": f"{elapsed_time:.2f} seconds"
    })

if __name__ == '__main__':
    app.run(debug=True)