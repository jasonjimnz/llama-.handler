import os
import subprocess
from flask import Flask, request, stream_with_context, render_template, jsonify
from llama_cpp import Llama

if not 'Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf' in os.listdir('./'):
    print("""RequirementError: 
Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf model is no downloaded

Getitng your model from: https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf
        """)

    proc = subprocess.run(
        [
            'wget',
            'https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf',
            '-O ',
            '/opt/app/Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf'
        ]
    )
    if proc.returncode == 1:
        raise Exception("Error downloading model")


def load_llm(model_path: str, **kwargs) -> Llama:
    params = {
        'model_path': model_path
    }
    if kwargs.get('gpu_layers'):
        # For using GPU
        params['n_gpu_layers'] = kwargs.get('gpu_layers')
    if kwargs.get('seed'):
        # For setting specific seed
        params['seed'] = kwargs.get('seed')
    if kwargs.get('n_ctx'):
        # For setting context window
        params['n_ctx'] = kwargs.get('n_ctx')
    llm = Llama(n_gpu_layers=0, **params)

    return llm


def query_llm(
        llm: Llama,
        question: str,
        stream: bool = False,
        max_tokens: int = 32,
        temperature: float = 0.8
):
    output = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream
    )
    return output


app = Flask(__name__)
model_file = "Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf"
model = f"{model_file}"
llm = load_llm(model_path=model, seed=1, n_ctx=1024)


@app.route('/')
def home():
    return render_template('chatbot.html')


@app.route('/ask_bot', methods=['POST'])
def ask_bot():
    data = request.json
    question = data.get('question')
    max_tokens = int(data.get('max_tokens', 1024))

    if not question:
        return jsonify({'error': 'Question param is required'}), 400

    resp = query_llm(
        llm=llm,
        question=question,
        stream=True,
        max_tokens=max_tokens
    )

    def response_coroutine():
        for c in resp:
            yield c['choices'][0]['delta'].get('content', '')

    return app.response_class(stream_with_context(response_coroutine()))


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
