# 1- Imports
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
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
    messages: Annotated[List, add_messages]

# 4- Create Node
def chat_node(state: Context):
    # Build conversation history from messages
    conversation = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            conversation += f"You: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation += f"Bot: {msg.content}\n"
    
    # Add current question
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
    
    # Return new AI message - it will be appended to messages automatically
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
st.title("🤖 Chatbot using LangGraph")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "threads" not in st.session_state:
    st.session_state.threads = {"1": []}  # Start with default thread
if "current_thread" not in st.session_state:
    st.session_state.current_thread = "1"

# Sidebar: Threads
st.sidebar.header("💬 Conversations")

# New Chat Button
if st.sidebar.button("➕ New Chat"):
    thread_id = str(len(st.session_state.threads) + 1)
    st.session_state.threads[thread_id] = []
    st.session_state.chat_history = []
    st.session_state.current_thread = thread_id
    st.rerun()

# Thread Selection
thread_id = st.sidebar.selectbox(
    "Select Thread",
    options=list(st.session_state.threads.keys()),
    index=list(st.session_state.threads.keys()).index(st.session_state.current_thread)
)

# Load selected thread if changed
if thread_id != st.session_state.current_thread:
    st.session_state.current_thread = thread_id
    st.session_state.chat_history = st.session_state.threads[thread_id]
    st.rerun()

# Display conversation history
st.markdown("---")
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**🧑 You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**🤖 Bot:** {msg.content}")

# Input section at bottom
st.markdown("---")
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input("Ask something:", key="user_input", label_visibility="collapsed", placeholder="Type your message here...")

with col2:
    send_button = st.button("Send", use_container_width=True)

# Handle user input
if send_button and user_input.strip():
    # Append user message
    user_message = HumanMessage(content=user_input)
    st.session_state.chat_history.append(user_message)
    
    # Invoke LangGraph workflow with FULL conversation history
    try:
        result = workflow.invoke(
            {
                "question": user_input,
                "messages": st.session_state.chat_history  # Pass full history
            },
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Extract bot response (last message in result)
        bot_message = result["messages"][-1]
        
        # Only append if it's not already in history (avoid duplicates)
        if not st.session_state.chat_history or st.session_state.chat_history[-1].content != bot_message.content:
            st.session_state.chat_history.append(bot_message)
        
        # Save updated history to threads
        st.session_state.threads[thread_id] = st.session_state.chat_history
        
        # Rerun to display new messages and clear input
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Clear chat button in sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Current Chat"):
    st.session_state.chat_history = []
    st.session_state.threads[thread_id] = []
    st.rerun()