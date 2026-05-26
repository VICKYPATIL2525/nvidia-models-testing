import json
import os
import sys
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA


APP_NAME = "Simple Coding Chatbot"
WORKSPACE_ROOT = Path.cwd().resolve()
LOG_FILE = WORKSPACE_ROOT / "chat_activity.json"
CONVERSATION_WINDOW = 8
MAX_AGENT_STEPS = 12
MAX_PARSE_RETRIES = 3

SYSTEM_PROMPT = """
You are a helpful coding assistant.
Answer software engineering questions clearly and practically.
When useful, include short code examples and explain tradeoffs.
""".strip()

AGENT_PROMPT = """
You are a coding file agent with access to local workspace tools.
Return exactly one JSON object per step. No markdown, no extra text.

Available actions:
1) {"action":"list_files","path":".","recursive":true}
2) {"action":"read_file","path":"main.py","start_line":1,"end_line":200}
3) {"action":"write_file","path":"notes.txt","content":"..."}
4) {"action":"replace_in_file","path":"main.py","old":"old text","new":"new text"}
5) {"action":"final","message":"summary for the user"}

Rules:
- Prefer small, precise edits.
- Read before editing.
- Keep paths inside the workspace only.
- When the task is complete, return action=final.
""".strip()


def now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


def estimate_tokens(text: str) -> int:
  if not text:
    return 0
  return ceil(len(text) / 4)


def normalize_content(content: Any) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts: List[str] = []
    for item in content:
      if isinstance(item, dict):
        parts.append(str(item.get("text", "")))
      else:
        parts.append(str(item))
    return "".join(parts)
  return str(content or "")


def resolve_workspace_path(path_text: str) -> Path:
  candidate = (WORKSPACE_ROOT / path_text).resolve()
  try:
    candidate.relative_to(WORKSPACE_ROOT)
  except ValueError as exc:
    raise ValueError("Path must stay inside current workspace.") from exc
  return candidate


def list_files(path_text: str = ".", recursive: bool = True) -> List[str]:
  base = resolve_workspace_path(path_text)
  if not base.exists():
    raise FileNotFoundError(f"Path does not exist: {path_text}")

  items: List[str] = []
  if base.is_file():
    return [str(base.relative_to(WORKSPACE_ROOT)).replace("\\", "/")]

  walker = base.rglob("*") if recursive else base.iterdir()
  for item in walker:
    if ".git" in item.parts or "myenv" in item.parts or "__pycache__" in item.parts:
      continue
    if recursive:
      if item.is_file():
        items.append(str(item.relative_to(WORKSPACE_ROOT)).replace("\\", "/"))
    else:
      suffix = "/" if item.is_dir() else ""
      items.append(str(item.relative_to(WORKSPACE_ROOT)).replace("\\", "/") + suffix)

  items.sort()
  return items


def read_file_lines(path_text: str, start_line: int = 1, end_line: int = 200) -> str:
  file_path = resolve_workspace_path(path_text)
  if not file_path.exists() or not file_path.is_file():
    raise FileNotFoundError(f"File not found: {path_text}")

  text = file_path.read_text(encoding="utf-8", errors="replace")
  lines = text.splitlines()
  start = max(1, start_line)
  end = min(len(lines), max(start, end_line))
  return "\n".join(lines[start - 1:end])


def write_file(path_text: str, content: str) -> str:
  file_path = resolve_workspace_path(path_text)
  file_path.parent.mkdir(parents=True, exist_ok=True)
  file_path.write_text(content, encoding="utf-8")
  return f"Wrote {path_text} ({len(content)} chars)."


def replace_in_file(path_text: str, old: str, new: str) -> str:
  if not old:
    raise ValueError("'old' must not be empty.")

  file_path = resolve_workspace_path(path_text)
  if not file_path.exists() or not file_path.is_file():
    raise FileNotFoundError(f"File not found: {path_text}")

  text = file_path.read_text(encoding="utf-8")
  count = text.count(old)
  if count == 0:
    raise ValueError("Text to replace was not found.")

  file_path.write_text(text.replace(old, new), encoding="utf-8")
  return f"Updated {path_text}. Replaced {count} occurrence(s)."


def parse_json_action(raw_text: str) -> Dict[str, Any]:
  text = raw_text.strip()
  if not text:
    raise ValueError("Empty model response")

  if text.startswith("```"):
    text_lines = text.splitlines()
    if len(text_lines) >= 3 and text_lines[-1].strip() == "```":
      text = "\n".join(text_lines[1:-1]).strip()

  try:
    parsed = json.loads(text)
    if isinstance(parsed, dict):
      return parsed
  except json.JSONDecodeError:
    pass

  start = text.find("{")
  end = text.rfind("}")
  if start >= 0 and end > start:
    parsed = json.loads(text[start:end + 1])
    if isinstance(parsed, dict):
      return parsed

  raise ValueError("Model response was not valid JSON")


def create_log_state(model_name: str, temperature: float) -> Dict[str, Any]:
  return {
    "meta": {
      "app": APP_NAME,
      "framework": "LangChain",
      "log_version": 1,
    },
    "session": {
      "started_at": now_iso(),
      "ended_at": None,
      "model": model_name,
      "temperature": temperature,
    },
    "totals": {
      "turns": 0,
      "input_messages": 0,
      "output_messages": 0,
      "input_characters": 0,
      "output_characters": 0,
      "input_tokens_estimated": 0,
      "output_tokens_estimated": 0,
    },
    "events": [],
    "chats": [],
  }


def load_log_state(model_name: str, temperature: float) -> Dict[str, Any]:
  if LOG_FILE.exists():
    try:
      data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
      if isinstance(data, dict):
        data.setdefault("events", [])
        data.setdefault("chats", [])
        data.setdefault("totals", {})
        data.setdefault("session", {})
        data.setdefault("meta", {})
        return data
    except json.JSONDecodeError:
      pass
  return create_log_state(model_name, temperature)


def save_log_state(state: Dict[str, Any]) -> None:
  LOG_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def append_event(state: Dict[str, Any], event_type: str, message: str) -> None:
  state["events"].append(
    {
      "timestamp": now_iso(),
      "type": event_type,
      "message": message,
    }
  )


def record_chat(state: Dict[str, Any], user_text: str, assistant_text: str) -> None:
  totals = state["totals"]
  totals["turns"] += 1
  totals["input_messages"] += 1
  totals["output_messages"] += 1
  totals["input_characters"] += len(user_text)
  totals["output_characters"] += len(assistant_text)
  totals["input_tokens_estimated"] += estimate_tokens(user_text)
  totals["output_tokens_estimated"] += estimate_tokens(assistant_text)

  state["chats"].append(
    {
      "id": totals["turns"],
      "timestamp": now_iso(),
      "input": user_text,
      "output": assistant_text,
      "counts": {
        "input_characters": len(user_text),
        "output_characters": len(assistant_text),
        "input_tokens_estimated": estimate_tokens(user_text),
        "output_tokens_estimated": estimate_tokens(assistant_text),
      },
    }
  )


def initialize_model() -> ChatNVIDIA:
  load_dotenv()
  model_name = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
  temperature = float(os.getenv("TEMPERATURE", "0.2"))
  return ChatNVIDIA(model=model_name, temperature=temperature)


def build_chat_messages(history: List[Any], user_text: str) -> List[Any]:
  messages: List[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
  messages.extend(history[-CONVERSATION_WINDOW * 2 :])
  messages.append(HumanMessage(content=user_text))
  return messages


def execute_agent_action(action: Dict[str, Any]) -> str:
  action_name = action.get("action")
  if action_name == "list_files":
    path_text = str(action.get("path", "."))
    recursive = bool(action.get("recursive", True))
    return json.dumps(list_files(path_text, recursive), indent=2)

  if action_name == "read_file":
    path_text = str(action.get("path", ""))
    start_line = int(action.get("start_line", 1))
    end_line = int(action.get("end_line", 200))
    return read_file_lines(path_text, start_line, end_line)

  if action_name == "write_file":
    path_text = str(action.get("path", ""))
    content = str(action.get("content", ""))
    return write_file(path_text, content)

  if action_name == "replace_in_file":
    path_text = str(action.get("path", ""))
    old = str(action.get("old", ""))
    new = str(action.get("new", ""))
    return replace_in_file(path_text, old, new)

  raise ValueError(f"Unsupported action: {action_name!r}")


def run_agent_task(chat_model: ChatNVIDIA, task_text: str) -> str:
  messages: List[Any] = [
    SystemMessage(content=AGENT_PROMPT),
    HumanMessage(
      content=(
        f"Workspace root: {WORKSPACE_ROOT}\n"
        f"Task: {task_text}\n"
        "Return the next action as JSON."
      )
    ),
  ]

  parse_failures = 0
  for _ in range(MAX_AGENT_STEPS):
    response = chat_model.invoke(messages)
    response_text = normalize_content(response.content)
    messages.append(AIMessage(content=response_text))

    try:
      action = parse_json_action(response_text)
    except Exception as exc:
      parse_failures += 1
      if parse_failures >= MAX_PARSE_RETRIES:
        return f"Agent failed to parse model action: {exc}"
      messages.append(
        HumanMessage(
          content=(
            "Your previous response was not valid JSON. "
            f"Error: {exc}. Return only one JSON object."
          )
        )
      )
      continue

    if action.get("action") == "final":
      return str(action.get("message", "Task complete."))

    try:
      tool_result = execute_agent_action(action)
    except Exception as exc:
      tool_result = f"Tool error: {exc}"

    messages.append(
      HumanMessage(
        content=(
          "Tool result:\n"
          f"{tool_result}\n\n"
          "Return the next JSON action."
        )
      )
    )

  return "Agent stopped after reaching the step limit."


def run_chat_loop() -> int:
  chat_model = initialize_model()
  model_name = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
  temperature = float(os.getenv("TEMPERATURE", "0.2"))
  state = load_log_state(model_name, temperature)
  state["session"].update({"model": model_name, "temperature": temperature})

  append_event(state, "startup", f"{APP_NAME} started.")
  save_log_state(state)

  history: List[Any] = []
  print(f"{APP_NAME} ready. Type /agent <task> for file operations, or exit to quit.")

  try:
    while True:
      try:
        user_text = input("you> ").strip()
      except EOFError:
        print()
        break

      if not user_text:
        continue

      if user_text.lower() in {"exit", "quit"}:
        break

      if user_text.startswith("/agent"):
        task_text = user_text[len("/agent"):].strip()
        if not task_text:
          print("agent> usage: /agent <task>")
          continue

        try:
          result_text = run_agent_task(chat_model, task_text)
          print(f"agent> {result_text}")
        except Exception as exc:
          result_text = f"Agent error: {exc}"
          print(f"agent> {result_text}")

        record_chat(state, user_text, result_text)
        save_log_state(state)
        continue

      messages = build_chat_messages(history, user_text)
      response = chat_model.invoke(messages)
      assistant_text = normalize_content(response.content).strip()
      if not assistant_text:
        assistant_text = "I did not get a usable response from the model."

      print(f"assistant> {assistant_text}")
      history.append(HumanMessage(content=user_text))
      history.append(AIMessage(content=assistant_text))
      record_chat(state, user_text, assistant_text)
      save_log_state(state)

  except KeyboardInterrupt:
    print()
  except Exception as exc:
    append_event(state, "error", str(exc))
    save_log_state(state)
    raise
  finally:
    append_event(state, "shutdown", "Chatbot stopped.")
    state["session"]["ended_at"] = now_iso()
    save_log_state(state)

  return 0


def main() -> int:
  load_dotenv()
  api_key = os.getenv("NVIDIA_API_KEY")
  if not api_key:
    print(
      "Missing NVIDIA_API_KEY. Add it to .env or set it in the terminal, then run again.",
      file=sys.stderr,
    )
    return 1

  return run_chat_loop()


if __name__ == "__main__":
  raise SystemExit(main())