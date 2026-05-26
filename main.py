import os
import sys

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

SYSTEM_PROMPT = """You are a helpful and knowledgeable assistant.
Answer any question clearly, concisely, and accurately.
If you don't know the answer, say so honestly."""

prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),("human", "{question}"),
])


def main() -> None:
    if not os.getenv("NVIDIA_API_KEY"):
        print("Error: NVIDIA_API_KEY not set. Add it to .env", file=sys.stderr)
        sys.exit(1)

    llm = ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
    )

    chain = prompt | llm

    history = []
    print("Chatbot ready. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        reply = chain.invoke({"question": question, "history": history})
        print(f"bot> {reply.content}\n")

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=reply.content))


if __name__ == "__main__":
    main()
