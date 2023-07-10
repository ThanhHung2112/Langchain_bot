import streamlit as st
import os
import openai
from io import StringIO

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryBufferMemory

from langchain.vectorstores import Chroma
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
import tempfile

import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
# openai.api_key = "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" 
global openai_api_key
openai_api_key = "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"   

os.environ['OPENAI_API_KEY'] = "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"


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]


chat = ChatOpenAI(temperature=0.0, max_tokens=20) 
memory = ConversationBufferWindowMemory(k=15)      
conversation = ConversationChain(
            llm=chat, 
            memory = memory,
            verbose=False, 
)


def reply(message, custom_style):
    style = """ in a funny \
    and joke tone
    """
    if len(custom_style) > 0: style = custom_style

    template_string = f"""You are talking with a person \
    replying to the message\
    with a style that is {style}. \
    the person just say: {message}.
    """
    
    prompt_template = ChatPromptTemplate.from_template(template_string)

    bot_messages = prompt_template.format_messages(
                    style= style,
                    text= message)
    
    response = conversation.predict(input=message)


    return response

def sumarization():
    pass

def document_question(question):
    pass
ask_about_doc = False
with st.sidebar:
    st.subheader("How do you want your bot reply to your message ?")
    custom_style = st.text_input("Tell me here", placeholder="joke tone")
    st.subheader("Wanna sumarization documentation?")
    
    uploaded_files = st.file_uploader("Upload file", accept_multiple_files=True)

    sidebar_empty = st.sidebar.empty()
    if uploaded_files:
        with sidebar_empty:

            smr_btn = st.sidebar.button('Summarize')
           
            ask_about_doc = st.sidebar.checkbox(label="Talk about the document ???")

            if smr_btn:
                sumarization()

container = st.container()
with container:
    st.title("🤖 AI ChatBot")
 

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if ask_about_doc==True:

    if prompt := st.chat_input("Ask anything about the doc"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = f"""Hi there {prompt}"""

        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
# React to user input
elif prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = reply(prompt,custom_style)
    if memory not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=15)
    # response=llm_chain.memory.chat_memory.add_user_message(prompt)
    st.write(memory.buffer)
     # #f"Echo: {prompt}" get_completion(template_string) #
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})