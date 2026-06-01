<div align="center">

# NVIDIA CLI Chatbot

**A simple terminal-based Q&A chatbot powered by free NVIDIA AI models and LangChain.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green?style=flat-square)
![NVIDIA](https://img.shields.io/badge/NVIDIA-AI%20Endpoints-76B900?style=flat-square&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

</div>

---

## Overview

A minimal CLI chatbot that runs in your terminal. Ask any question and get an answer. It remembers the conversation so follow-up questions work naturally. Built with the **free NVIDIA AI Endpoints API** and **LangChain** in a single Python file.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| [NVIDIA AI Endpoints](https://build.nvidia.com) | Free hosted LLM API |
| [LangChain](https://python.langchain.com) | LLM wrapper and message handling |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Load API key from `.env` |

---

## Setup

**1. Get a free NVIDIA API key**

Sign up at [build.nvidia.com](https://build.nvidia.com), pick any free model, and copy the API key.

**2. Clone and install**

```bash
git clone https://github.com/your-username/nvidia-models-testing.git
cd nvidia-models-testing

python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # Mac / Linux

pip install -r requirements.txt
```

**3. Create a `.env` file**

```env
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
NVIDIA_MODEL=meta/llama-3.1-8b-instruct
TEMPERATURE=0.2
```

> Never commit `.env` to git — it is already in `.gitignore`.

---

## Run

```bash
python main.py
```

---

## Example

```
Chatbot ready. Type 'exit' to quit.

you> what is a large language model?
bot> A large language model (LLM) is a type of AI trained on massive amounts
     of text data to understand and generate human language ...

you> give me a simple python example of a class
bot> Sure! Here's a simple example ...

you> exit
Bye!
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | *(required)* | Your free NVIDIA API key |
| `NVIDIA_MODEL` | `meta/llama-3.1-8b-instruct` | Model to use |
| `TEMPERATURE` | `0.2` | Creativity — 0.0 focused, 1.0 creative |

### Available free models

| Model ID | Best For |
|---|---|
| `meta/llama-3.1-8b-instruct` | General Q&A, fast (default) |
| `meta/llama-3.1-70b-instruct` | Complex reasoning |
| `mistralai/mistral-7b-instruct-v0.3` | Concise responses |
| `microsoft/phi-3-mini-128k-instruct` | Long context |

Browse all free models at [build.nvidia.com/explore/reasoning](https://build.nvidia.com/explore/reasoning).

If you get a `404 Not Found` error, the selected model may be unavailable or deprecated for your account/region. Try another free model from the list above.

---

## Project Structure

```
nvidia-models-testing/
├── main.py           # entire chatbot in one file
├── requirements.txt  # dependencies
├── .env              # your API key (git-ignored)
├── .gitignore
└── README.md
```
