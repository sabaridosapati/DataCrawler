# gpu_services/llm_service/main.py

# This file is intentionally simple. The primary goal is to launch the vLLM server.
# We will do this via the Dockerfile's CMD instruction in the docker-compose.yml
# for maximum flexibility.

# This Python file is a placeholder to show the structure and can be used for
# future extensions, such as adding custom API endpoints if needed.

from fastapi import FastAPI

app = FastAPI(
    title="High-Performance LLM Service",
    description="Hosts a local Hugging Face model using the vLLM engine."
)

@app.get("/health")
async def health_check():
    """
    A simple health check endpoint.
    In a real-world scenario, you could add a check here to verify
    that the vLLM engine is loaded and healthy.
    """
    # Note: This health check is separate from the vLLM OpenAI API.
    # It runs on the same server but is not part of the core vLLM functionality.
    return {"status": "ok", "model_to_be_served": "google/gemma-3n-E2B-it"}