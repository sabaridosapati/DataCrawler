#!/bin/bash
# This script ensures the Ollama server starts and our custom model is created.
# The 'set -e' command will make the script exit immediately if any command fails.
set -e

# 1. Start the main Ollama server process in the background.
# The '&' symbol sends the process to the background.
echo "Starting Ollama server..."
ollama serve &

# 2. Capture the Process ID (PID) of the background server.
# We need this to keep the container running.
pid=$!

# 3. Wait for a few seconds to give the server time to initialize properly
# before we try to create a model on it.
echo "Waiting for Ollama server to initialize (5 seconds)..."
sleep 5

# 4. Create our custom 'gemma-3n' model endpoint using the Modelfile.
# This command tells Ollama to create a new model named 'gemma-3n'
# based on the instructions in the /app/Modelfile.
# Ollama will automatically download the base 'gemma:2b-instruct' model if needed.
echo "Creating the 'gemma-3n' model endpoint from Modelfile..."
ollama create gemma-3n -f /app/Modelfile

echo "----------------------------------------------------"
echo "✅ Ollama is up and running with the gemma-3n model."
echo "----------------------------------------------------"

# 5. Wait for the background server process (pid) to exit.
# This is the command that keeps the container alive. The 'ollama serve'
# process will run indefinitely, so this 'wait' command will also wait indefinitely.
wait $pid