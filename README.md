# nvidia-models-testing

Simple command-line coding chatbot built with LangChain and NVIDIA AI endpoints.

## Features

- Uses LangChain chat model wrapper for NVIDIA endpoints
- Writes full session activity to a JSON log file
- Reads configuration from environment variables or a .env file
- Focused on coding question-and-answer use cases

## Project Structure

- main.py: chatbot app entrypoint
- requirements.txt: Python dependencies
- .env: local secrets and config (not committed)
- chat_activity.json: generated runtime logs and totals

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

	pip install -r requirements.txt

3. Create or update your .env file with:

	NVIDIA_API_KEY=your_api_key_here
	NVIDIA_MODEL=meta/llama-3.1-8b-instruct
	TEMPERATURE=0.2

## Run

python main.py

Then ask coding questions directly in the terminal.
Type exit or quit to stop.

## Agent Mode (Read/Edit/Save Files)

You can run file-editing tasks with:

/agent <your task>

Examples:

- /agent list all python files in this project
- /agent read main.py and explain what to improve
- /agent create utils.py with a function to validate email
- /agent replace temperature default from 0.2 to 0.1 in main.py

Supported file actions in workspace:

- List files
- Read file ranges
- Write files
- Replace text in files

Safety:

- Agent actions are restricted to the current workspace directory.

## Logging

The app creates and updates chat_activity.json automatically.

It tracks:

- Per-turn input and output text
- Number of messages
- Character counts for input and output
- Estimated input and output tokens
- Reported token usage from the model (if returned)
- Session events like startup, errors, and shutdown

## Notes

- Do not hardcode API keys in source code.
- You can change the model by updating NVIDIA_MODEL in .env.
