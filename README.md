# openvino-api-server
my personal attempt to openai API compatible openvino server



```
python -m venv venv
pip install openvino-genai==2025.0.0
pip install optimum[openvino,nncf] torchvision evaluate openai fastapi uvicorn tiktoken gradio
```
this will install a lot of packages

```
Installing collected packages: sentencepiece, pytz, pydub, py-cpuinfo, mpmath, jstyleson, grapheme, xxhash, wrapt, websockets, urllib3, tzdata, typing-extensions, tomlkit, threadpoolctl, tabulate, sympy, sniffio, six, shellingham, semantic-version, safetensors, ruff, rpds-py, regex, pyyaml, python-multipart, pyparsing, pygments, pyarrow, psutil, protobuf, propcache, pillow, orjson, numpy, ninja, networkx, natsort, multidict, mdurl, markupsafe, kiwisolver, joblib, jiter, idna, h11, groovy, fsspec, frozenlist, fonttools, filelock, ffmpy, distro, dill, cycler, colorama, charset-normalizer, certifi, attrs, annotated-types, aiohappyeyeballs, aiofiles, about-time, yarl, tqdm, scipy, requests, referencing, python-dateutil, pydot, pydantic-core, onnx, multiprocess, markdown-it-py, jinja2, httpcore, Deprecated, contourpy, cma, click, autograd, anyio, alive-progress, aiosignal, uvicorn, torch, tiktoken, starlette, scikit-learn, rich, pydantic, pandas, matplotlib, jsonschema-specifications, huggingface-hub, httpx, aiohttp, typer, torchvision, tokenizers, safehttpx, pymoo, openai, jsonschema, gradio-client, fastapi, transformers, nncf, gradio, datasets, optimum, evaluate, optimum-intel
```

Set HF-Mirror for Chinese users
```
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

Login with your HF TOKEN and download the model
```
huggingface-cli login
huggingface-cli download --resume-download boysbytes/DeepSeek-R1-Distill-Qwen-1.5B-openvino-4bit --local-dir DeepSeek-R1-Distill-Qwen-1.5B-openvino-4bit
```
---

Created with Cluade Sonnet 3.7

Read the entire MakingOF and screenshots from [here](https://github.com/fabiomatricardi/openvino-api-server/raw/main/MakingOF/readme.md)

Made with <img src='https://github.com/fabiomatricardi/openvino-api-server/raw/main/MakingOF/000.png' height=20>

## Read The Docs

# OpenVINO GenAI API Server Documentation

## Introduction

The OpenVINO GenAI API Server is a FastAPI-based implementation that provides an OpenAI-compatible API interface for running local language models with OpenVINO. This server enables applications designed for OpenAI's API to work with locally-hosted models, specifically targeting the chat completions endpoint with streaming capabilities.

## Features

- OpenAI API-compatible interface (`/v1/chat/completions`)
- Support for both streaming and non-streaming responses
- Built with FastAPI for high performance and async support
- Powered by OpenVINO GenAI for efficient model inference
- Configurable model settings including temperature and token limits

## Installation

### Prerequisites

- Python 3.11 - works BEST with OpenVINO GenAI, without breaking the dependencie
- OpenVINO GenAI library
- FastAPI and Uvicorn

### Setup

1. Install the required dependencies:

```bash
pip install -U fastapi uvicorn openvino_genai tiktoken gradio openai
# to convert the models
pip install optimum[openvino,nncf] torchvision evaluate 
```

2. Download the SmolLM2-360M-Instruct model (or your preferred model) and convert it to OpenVINO format if necessary.

3. Clone or copy the server code to `app.py`.

## Server Configuration

The server is configured to use the SmolLM2-360M-Instruct model by default. You can modify the model configuration by editing the following variables in the code:

```python
model_path = "SmolLM2-360M-Instruct-openvino-8bit"  # Path to your model
device = "GPU"  # Inference device (CPU, GPU, etc.)
```

## Running the Server

Start the server by running:

```bash
python app.py
```

Or with uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at http://localhost:8000.

## API Endpoints

### Chat Completions

```
POST /v1/chat/completions
```

This endpoint generates a chat completion response based on the provided messages.

#### Request Body

```json
{
  "model": "SmolLM2-360M-Instruct",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "temperature": 0.7,
  "top_p": 1.0,
  "n": 1,
  "stream": false,
  "max_tokens": 900,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "stop": null
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | string | *required* | The model to use (note: this server always uses the configured model regardless of this parameter) |
| messages | array | *required* | An array of message objects with role and content |
| temperature | float | 0.7 | Controls randomness in the output |
| top_p | float | 1.0 | Controls diversity via nucleus sampling |
| n | integer | 1 | Number of completions to generate (only 1 is currently supported) |
| stream | boolean | false | Whether to stream the response |
| max_tokens | integer | 900 | Maximum number of tokens to generate |
| presence_penalty | float | 0.0 | Penalty for token repetition (not currently implemented) |
| frequency_penalty | float | 0.0 | Penalty for token frequency (not currently implemented) |
| stop | string/array | null | Stop sequence to end generation (not currently implemented) |

#### Response (Non-streaming)

```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1678048400,
  "model": "SmolLM2-360M-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

#### Streaming Response

For streaming responses (`stream: true`), the server sends a series of server-sent events (SSE) in the following format:

```
data: {"id":"chatcmpl-1234567890","object":"chat.completion.chunk","created":1678048400,"model":"SmolLM2-360M-Instruct","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-1234567890","object":"chat.completion.chunk","created":1678048400,"model":"SmolLM2-360M-Instruct","choices":[{"index":0,"delta":{"content":"I"},"finish_reason":null}]}

data: {"id":"chatcmpl-1234567890","object":"chat.completion.chunk","created":1678048400,"model":"SmolLM2-360M-Instruct","choices":[{"index":0,"delta":{"content":"'m"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-1234567890","object":"chat.completion.chunk","created":1678048400,"model":"SmolLM2-360M-Instruct","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}]}

data: [DONE]
```

## Client Integration

### Python OpenAI Client (v1.0.0+)

```python
from openai import OpenAI

# Initialize the client with your base URL
client = OpenAI(
    api_key="not-needed",  # Can be any string when using your server
    base_url="http://localhost:8000/v1",
)

# For streaming
response = client.chat.completions.create(
    model="SmolLM2-360M-Instruct",
    messages=[
        {"role": "user", "content": "Tell me about quantum computing"}
    ],
    stream=True
)

# Process the streaming response
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

# For non-streaming
response = client.chat.completions.create(
    model="SmolLM2-360M-Instruct",
    messages=[
        {"role": "user", "content": "Tell me about quantum computing"}
    ],
    stream=False
)

print(response.choices[0].message.content)
```

### Node.js OpenAI Client

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: 'not-needed',
  baseURL: 'http://localhost:8000/v1',
});

async function main() {
  // Non-streaming example
  const completion = await openai.chat.completions.create({
    model: 'SmolLM2-360M-Instruct',
    messages: [{ role: 'user', content: 'Hello, how are you?' }],
  });

  console.log(completion.choices[0].message.content);

  // Streaming example
  const stream = await openai.chat.completions.create({
    model: 'SmolLM2-360M-Instruct',
    messages: [{ role: 'user', content: 'Tell me a story' }],
    stream: true,
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || '');
  }
}

main();
```

### CURL

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM2-360M-Instruct",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

For PowerShell:

```powershell
Invoke-WebRequest -Uri "http://localhost:8000/v1/chat/completions" -Method Post -ContentType "application/json" -Body '{"model": "SmolLM2-360M-Instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Code Architecture

The server implementation consists of the following main components:

1. **Model Initialization**: Sets up the OpenVINO GenAI model and tokenizer
2. **API Models**: Pydantic models for request and response validation
3. **Streaming Handler**: Custom streaming implementation for token-by-token output
4. **Endpoint Handler**: FastAPI route for chat completions
5. **Async Generation**: Asynchronous handling of model inference

## Limitations

- The server currently only implements the `/v1/chat/completions` endpoint
- Some parameters (like `presence_penalty`, `frequency_penalty`, and `stop`) are accepted but not currently implemented
- Token count is an approximation based on word count, not the actual token count
- The server always uses the configured model, regardless of the model name in the request

## Troubleshooting

### Connection Refused Errors

If you see "Connection refused" errors when connecting to the server:

1. Verify the server is running and check for any error messages
2. Ensure the port is not in use by another application
3. Check if you're using the correct host and port in your client

### Model Loading Issues

If the model fails to load:

1. Verify the model path is correct
2. Ensure OpenVINO is properly installed with appropriate drivers
3. Check if you have sufficient memory/VRAM for the model

## Advanced Configuration

### Custom Port

To run the server on a different port, change the port parameter in the `uvicorn.run()` call:

```python
uvicorn.run("app:app", host="0.0.0.0", port=9000, reload=True)
```

### Production Deployment

For production deployment, disable the reload feature and consider using Gunicorn with Uvicorn workers:

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

### Model Parameters

To adjust model parameters, modify the generation settings in the code:

```python
# Non-streaming response
response_text = pipe.generate(
    model_inputs, 
    max_new_tokens=request.max_tokens,
    temperature=request.temperature,  # Add this parameter if supported
    top_p=request.top_p              # Add this parameter if supported
)
```

## License

[MIT License](https://opensource.org/licenses/MIT)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
