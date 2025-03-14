from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama


import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a cancer biologist."),
        ("user", "Question:{question}")
    ]
)

def generate_response(question,llm,temparature, max_tokens):

    llm=Ollama(model=llm)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

## #Title of the app
st.title("Enhanced Q&A Chatbot With Ollama")

## select the ollama model
llm=st.sidebar.selectbox("select ollama models",["orca-mini"])

## Adjust response parameters
temparature=st.sidebar.slider("Temparature", min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## main interface for user input
st.write("Ask any question, and I'll help you out.")
user_input=st.text_input("you:")

if user_input:
    response=generate_response(user_input,llm,temparature,max_tokens)
    st.write(response)

else:
    st.write("Please provide the user input")

