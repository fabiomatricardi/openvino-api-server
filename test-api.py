from openai import OpenAI

# Initialize the client with your base URL
client = OpenAI(
    api_key="not-needed",  # Can be any string when using your server
    base_url="http://localhost:8000/v1",
)

# For streaming
response = client.chat.completions.create(
    model="SmolLM2-360M-Instruct",  # Model name can be anything, server uses configured model
    messages=[
        {"role": "user", "content": "Tell me about quantum computing"}
    ],
    stream=True
)

# Process the streaming response
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

"""
# For non-streaming
response = client.chat.completions.create(
    model="SmolLM2-360M-Instruct",
    messages=[
        {"role": "user", "content": "Tell me about quantum computing"}
    ],
    stream=False
)

print(response.choices[0].message.content)
"""