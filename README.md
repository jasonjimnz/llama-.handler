# Llama test API

## Installation
I recommend using Anaconda for easy installation, using virtualenv is OK
but Anaconda helps you with some C++ dependencies specially on Windows

With Anaconda run:
```bash
conda install conda-forge::llama-cpp-python flask
```

Without Anaconda you will need extra config in order to
get a proper build of llama.cpp:

without Anaconda run:
```bash
# Linux and Mac
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
  pip install llama-cpp-python
```

OR

```bash
# Windows
$env:CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
pip install llama-cpp-python flask
```

### Important
On Windows make sure that you have VS Studio tools installed for building llama-cpp dependencies, 
depending on your GPU maybe you must do additional steps.

But if you want to avoid that, you can just run this Docker image

[Docker image](https://hub.docker.com/repository/docker/jasonjimnz/flask-llama/general)

## Usage
First start your Anaconda environment or your Virtual Environment

Consider downloading the model (the code can download it by itself but I always recommend to do it manually)

```shell
wget https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf -O ./Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf
```

Once you have the model downloaded (Aprox. 4GB), start the Flask server:

```shell
python api.py
```

It will run a Flask server with `0.0.0.0` host so, every computer in the
network should be able to visit the site, for testing in local just call
the following URL:  `http://localhost:5000`


## NOTES
- The context is limited by default, you can change it but if you increase it
    you will need more resources
- The max_token is also limited by default, same as context
- It does not handle a full conversation, every message will require whole context
    just for saving resources, more input, more tokens you need to handle it
- You can download any other model from HuggingFace and load it, I use Meta Llama 3.1 8B
    because fits perfectly in 8GB (first tests were done in a 2Core 8GB RAM VirtualMachine)
- Frontend can be better, but you also have the /ask_bot endpoint exposed, so you can do
  your own integration based on this example