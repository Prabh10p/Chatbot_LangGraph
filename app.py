# 1- Imports
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import BaseMessage, add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
import streamlit as st
import sqlite3

load_dotenv()

# 2- Define Model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational",
    temperature=0.8,
)
model = ChatHuggingFace(llm=llm)

# 3- Define State
class Context(TypedDict):
    question: str
    messages: Annotated[List[BaseMessage], add_messages]

# 4- Create Node
def chat_node(state: Context):
    conversation = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            conversation += f"You: {msg.content}\n"
        else:
            conversation += f"Bot: {msg.content}\n"

    conversation += f"You: {state['question']}\nBot:"

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

# 5- Define Graph
graph = StateGraph(Context)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# 6- Persistence
conn = sqlite3.connect("memory.db", check_same_thread=False)
checkpoint = SqliteSaver(conn=conn)
workflow = graph.compile(checkpointer=checkpoint)

# 7- Streamlit App
st.title("Chatbot using LangGraph")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "threads" not in st.session_state:
    st.session_state.threads = {}  # store past thread histories by thread_id

# Sidebar: Threads
st.sidebar.header("Past Conversations")
if st.sidebar.button("New Chat"):
    thread_id = str(len(st.session_state.threads) + 1)
    st.session_state.threads[thread_id] = []
    st.session_state.chat_history = []
else:
    thread_id = st.sidebar.selectbox(
        "Select Thread",
        options=list(st.session_state.threads.keys()) or ["1"]
    )
    if thread_id not in st.session_state.threads:
        st.session_state.threads[thread_id] = []

# Load selected thread
st.session_state.chat_history = st.session_state.threads.get(thread_id, [])

# Input
user_input = st.text_input("Ask something:")
if st.button("Send") and user_input.strip():
    # Append user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Invoke LangGraph workflow
    result = workflow.invoke({"question": user_input}, config={"configurable": {"thread_id": thread_id}})
    bot_message = result["messages"][-1]
    st.session_state.chat_history.append(bot_message)

    # Save updated history to threads
    st.session_state.threads[thread_id] = st.session_state.chat_history

# Display conversation
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**🧑 You:** {msg.content}")
    else:
        st.markdown(f"**🤖 Bot:** {msg.content}")
