# this code is for demonstration purposes only and is not production-ready.
# this is a simple chatbot that uses the NVIDIA LLM API to answer questions.
# these are the necessary imports for the code to run. You may need to install the required packages using pip:
# pip install python-dotenv langchain-core langchain-nvidia-ai-endpoints
import os
import sys

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
# load environment variables from .env file
load_dotenv()
# define the system prompt that will be used to guide the chatbot's behavior. This prompt tells the chatbot to be helpful, knowledgeable, and honest about its limitations.
SYSTEM_PROMPT = """You are a helpful and knowledgeable assistant.
Answer any question clearly, concisely, and accurately.
If you don't know the answer, say so honestly."""
# create a chat prompt template that includes the system prompt, a placeholder for the conversation history, and a placeholder for the user's question. The conversation history will be updated after each interaction to provide context for the chatbot's responses.
prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),("human", "{question}"),
])

# the main function that runs the chatbot. It checks for the NVIDIA_API_KEY environment variable, initializes the ChatNVIDIA model, and enters a loop to interact with the user. The user's input is processed, and the chatbot's response is generated and displayed. The conversation history is updated after each interaction to maintain context.
def main():
    if not os.getenv("NVIDIA_API_KEY"): # check if the NVIDIA_API_KEY environment variable is set, which is required to access the NVIDIA LLM API. If it's not set, print an error message and exit the program.
        print("Error: NVIDIA_API_KEY not set. Add it to .env", file=sys.stderr)
        sys.exit(1)#sys.exit(1) is used to exit the program with a status code of 1, which indicates that an error occurred.
# initialize the ChatNVIDIA model with the specified model name and temperature. The model name can be set using the NVIDIA_MODEL environment variable, and the temperature can be set using the TEMPERATURE environment variable. If these variables are not set, default values will be used.
    llm = ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-r1"),# we are using the deepseek-r1 model from NVIDIA, which is a powerful language model designed for various natural language processing tasks.
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
    )

    chain = prompt | llm # create a chain that combines the chat prompt template and the ChatNVIDIA model. This allows us to generate responses based on the user's input and the conversation history.
# initialize an empty list to store the conversation history. This history will be updated after each interaction to provide context for the chatbot's responses.
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
# invoke the chain with the user's question and the conversation history. The chain will generate a response based on the input and the context provided by the history. The response is then printed to the console.
        reply = chain.invoke({"question": question, "history": history})
        print(f"bot> {reply.content}\n")

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=reply.content))

# the if __name__ == "__main__": block is a common Python idiom that checks if the script is being run directly (as the main program) rather than imported as a module. If the script is run directly, the main() function will be called to start the chatbot. 
if __name__ == "__main__":
    main()
