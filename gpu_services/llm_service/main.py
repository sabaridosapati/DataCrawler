# gpu_services/llm_service/main.py

# This file is intentionally simple. The primary goal is to launch the vLLM server.
# We will do this via the Dockerfile's CMD instruction for robustness.
# This Python file is a placeholder to show the structure and can be used for future extensions.

from fastapi import FastAPI

app = FastAPI(
    title="Local LLM Service",
    description="Hosts a local Hugging Face model using vLLM."
)

@app.get("/health")
async def health_check():
    # In a real scenario, you'd add a check to see if the vLLM engine is healthy.
    return {"status": "ok", "model": "google/gemma-3n-E2B"}