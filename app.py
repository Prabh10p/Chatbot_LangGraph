# 1- Importing Libraries
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import BaseMessage, add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
import streamlit as st
import sqlite3

load_dotenv()

# 2- Defining a Model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational",
    temperature=0.8,
)

model = ChatHuggingFace(llm=llm)

# 3- Defining State
class Context(TypedDict):
    question: str
    messages: Annotated[List[BaseMessage], add_messages]

# 4- Creating Node
def chat_node(state: Context):
    # Build conversation from all previous messages
    conversation = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            conversation += f"You: {msg.content}\n"
        else:  # AIMessage
            conversation += f"Bot: {msg.content}\n"

    # Add current question
    conversation += f"You: {state['question']}\nBot:"

    # Prompt instructing the bot to use prior numeric context
    prompt = PromptTemplate(
        template=(
            "You are an intelligent chatbot. Continue the conversation below.\n"
            "If previous messages contain numeric answers, use them for follow-up calculations.\n\n"
            "{conversation}"
        ),
        input_variables=["conversation"]
    )

    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"conversation": conversation})

    return {"messages": [AIMessage(content=response)]}



# 5- Defining Graph
graph = StateGraph(Context)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# 6- Persistence
conn= sqlite3.connect(database="memory.db",check_same_thread=False)
checkpoint = SqliteSaver(conn=conn)
workflow = graph.compile(checkpointer=checkpoint)

# 7- Config
thread_id = "1"
config = {"configurable": {"thread_id": thread_id}}

# 8- Streamlit App
st.title("Chatbot using LangGraph")
st.sidebar.button("New Chat")
st.sidebar.header("Past Conversation")

# Initialize session state for chat history


def load_conversation(thread_id1):
      return workflow.get_state(config={'configurable':{'thread_id':thread_id1}}.values['messages'])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if 'chat_thread' not in st.session_state:
      st.session_state["chat_thread"] = []

for thread_id in st.session_state["chat_thread"]:
      if st.sidebar.button(str(thread_id)):
           message = load_conversation(thread_id) 

user_input = st.text_input("Ask something:")

if st.button("Analyze"):
  if user_input.strip():
    # Add user message to session state
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Invoke workflow; LangGraph memory automatically includes prior messages
    result = workflow.invoke(
        {"question": user_input},
        config=config
    )

    # Add bot response to session state
    bot_message = result["messages"][-1]
    st.session_state.chat_history.append(bot_message)

# Display full conversation thread
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**🧑 You:** {msg.content}")
    else:
        st.markdown(f"**🤖 Bot:** {msg.content}")
