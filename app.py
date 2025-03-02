import sys
import asyncio
import time
import json
from typing import List, Optional, Union, Dict, Any, AsyncIterator

import openvino_genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Initialize OpenVINO GenAI with your configuration
model_path = "SmolLM2-360M-Instruct-openvino-8bit"
device = "GPU"
tokenizer = openvino_genai.Tokenizer(model_path)
pipe = openvino_genai.LLMPipeline(model_path, tokenizer=tokenizer, device=device)

app = FastAPI(title="OpenAI Compatible API Server")

# Define request and response models to match OpenAI's API structure
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 900
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None

class CompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]

# Custom streaming handler for OpenVINO GenAI
class APIStreamer:
    def __init__(self, response_queue):
        self.response_queue = response_queue
        self.full_response = ""
    
    def __call__(self, subword):
        self.full_response += subword
        self.response_queue.put_nowait(subword)
        # Return False to continue generation
        return False

# Stream generator for streaming responses
async def generate_stream(request: ChatCompletionRequest) -> AsyncIterator[str]:
    # Generate a response ID
    response_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())
    
    # Send the first chunk with role
    first_chunk = ChatCompletionStreamResponse(
        id=response_id,
        created=created_time,
        model=request.model,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None
            )
        ]
    )
    yield f"data: {json.dumps(first_chunk.dict())}\n\n"
    
    # Create a queue for the streamer
    queue = asyncio.Queue()
    streamer = APIStreamer(queue)
    
    # Run the model generation in a separate task
    generation_task = asyncio.create_task(generate_async(request.messages, request.max_tokens, streamer))
    
    # Stream the output tokens
    try:
        while True:
            try:
                # Get the next token with a timeout
                token = await asyncio.wait_for(queue.get(), timeout=0.1)
                
                chunk = ChatCompletionStreamResponse(
                    id=response_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(content=token),
                            finish_reason=None
                        )
                    ]
                )
                
                yield f"data: {json.dumps(chunk.dict())}\n\n"
                
                # Mark the task as done
                queue.task_done()
                
            except asyncio.TimeoutError:
                # Check if generation is complete
                if generation_task.done():
                    break
    
    finally:
        # Send the final chunk
        final_chunk = ChatCompletionStreamResponse(
            id=response_id,
            created=created_time,
            model=request.model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=""),
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {json.dumps(final_chunk.dict())}\n\n"
        
        # End the stream
        yield "data: [DONE]\n\n"

async def generate_async(messages, max_tokens, streamer):
    """Run the model generation in an asyncio-friendly way"""
    loop = asyncio.get_event_loop()
    
    def _generate():
        history = [{"role": m.role, "content": m.content} for m in messages]
        model_inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True)
        answer = pipe.generate(model_inputs, max_new_tokens=max_tokens, streamer=streamer)
        return answer
    
    # Run in a thread pool to avoid blocking the event loop
    return await loop.run_in_executor(None, _generate)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Check if streaming is requested
        if request.stream:
            return StreamingResponse(
                generate_stream(request),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        history = [{"role": m.role, "content": m.content} for m in request.messages]
        model_inputs = tokenizer.apply_chat_template(history, add_generation_prompt=True)
        
        # Count tokens for usage metrics (approximation)
        input_tokens = len(model_inputs.split())
        
        # Generate without streaming
        response_text = pipe.generate(model_inputs, max_new_tokens=request.max_tokens)
        
        # Count output tokens (approximation)
        output_tokens = len(response_text.split())
        
        # Create the response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)