import os
os.system("pip install --upgrade pip")
import re
import time
import io

from io import StringIO
from typing import Any, Dict, List
#Modules to Import
import openai
import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader

import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryBufferMemory

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
import tempfile

import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def parse_pdf (file: io.BytesIO)-> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:

        text = page.extract_text()
        #Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", "\1\2", text)
        # Fix newlines in the middle of sentences 
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        #Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        
        output.append(text)
    return output

@st.cache_data
def text_to_docs(text: str) -> List [Document]:

    """Converts a string or list of strings to a list of Documents with metadata,"""

    if isinstance(text, str):
        #Take a single string as one page 
        text = [text]
    page_docs = [Document (page_content=page) for page in text]
    # Add page numbers as metadata 
    for i, doc in enumerate(page_docs): 

        doc.metadata["page"] = 1 + 1
    # Split pages into chunks 
    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter( 
            chunk_size=2500, 
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": 1}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}" 
            doc_chunks.append(doc)
    return doc_chunks
def tool(index):

    qa = RetrievalQA.from_chain_type(
            llm = OpenAI(openai_api_key = api),
            chain_type = "stuff",
            retriever = index.as_retriever()
        )
        # our tool
    tools = [
        Tool(
            name="State of Union QA System",
            func=qa.run,
            description="Useful for when you need to answer questions about the aspects asked.\
            Input may be a partial or fully formed question,\
                it also can be about some things else, use the chat history to reply the questions"
        )
    ]
    return tools,qa
def process(kind, tools, qa):
    if kind == "Sumarized":
        prefix=""""Have a conversation with a human, answering the human questions as best you can based on the context and memory available. \
                You have access to a single tool:"""
        suffix="""Begin!"
        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

    elif kind == "Chat":
        prefix=""""Have a conversation with a human, answering the human questions as best you can \
            You have access to a single tool:"""

        suffix="""Begin!"
        {chat_history}
        the human just say: {input}
        {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key ="chat_history")
            #Chain
            # ZeroShotAgent
    llm_chain = LLMChain(
        llm=OpenAI(
        temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"
        ),
        prompt=prompt,
    )
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True) 
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
    )
    return agent_chain,llm_chain


option = st.sidebar.selectbox(
    'What do you want to ?',
    ('Sumarization','Chat'))

api = st.sidebar.text_input(
                    "Open api key",
                    type="password",
                    placeholder="sk-",
                    help="https://platform.openai.com/account/api-keys",
                )
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
# openai.api_key = "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" 
global openai_api_key
openai_api_key = api  
os.environ['OPENAI_API_KEY'] = openai_api_key
uploaded_file = st.sidebar.file_uploader(":blue[Upload]", type=["pdf"])
global agent_chain,llm_chain
if api:
    if option == "Sumarization":
    
        if uploaded_file:

            doc = parse_pdf(uploaded_file)

            pages = text_to_docs(doc)
            # pages
            if pages:
                with st.expander('Show page contents', expanded=False):
                    page_sel =st.number_input(
                        label="selected page", min_value=1, max_value=len(pages), step=1
                    )
                    st.write(pages[page_sel-1])
                    
                    
                    embeddings = OpenAIEmbeddings(openai_api_key = api)
                    # Indexing
                    # Save in a Vector DB_
                    with st.spinner("It's indexing. .."):
                        index = FAISS.from_documents(pages, embeddings)
                    
                    tools,qa = tool(index)
                    prefix=""""Have a conversation with a human, answering the human questions as best you can based on the context and memory available. \
                        He may ask some not about the context but just answer the the question with a short sentence"""
                    suffix="""Begin!"
                    {chat_history}
                    Question: {input}
                    {agent_scratchpad}"""
                    prompt = ZeroShotAgent.create_prompt(
                        tools,
                        prefix=prefix,
                        suffix=suffix,
                        input_variables=["input", "chat_history", "agent_scratchpad"],
                    )

                    if "memory" not in st.session_state:
                        st.session_state.memory = ConversationBufferMemory(memory_key ="chat_history")
                            #Chain
                            # ZeroShotAgent
                    llm_chain = LLMChain(
                        llm=OpenAI(
                        temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"
                        ),
                        prompt=prompt,
                    )
                    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True) 
                    agent_chain = AgentExecutor.from_agent_and_tools(
                        agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
                    )
                    # agent_chain,llm_chain = process("Sumarized",tools, qa)


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


                if query := st.chat_input("Hey yo !!! Wazzups!"):
                
                    st.chat_message("user").markdown(query)
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": query})
                
                    # response=llm_chain.memory.chat_memory.add_user_message(prompt)

                    if len(api) == 0:
                        response = f"""I will answer the question "{query}" if you give the API key"""
                        # st.write(response)
                        # #f"Echo: {prompt}" get_completion(template_string) #
                        # Display assistant response in chat message container
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    else:
                            
                        with st.spinner("It's indexing. .."):
                            
                            response = agent_chain.run(query)
                        # st.write(response)
                        # #f"Echo: {prompt}" get_completion(template_string) #
                        # Display assistant response in chat message container
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        # with st.expander("History/Memory"):
                        #     st.write(st.session_state.memory)

    elif option == "Chat":

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
        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("It's indexing. .."):
                response = reply(prompt,custom_style)
            # with st.spinner("It's indexing. .."):
            #     tools,qa = tool()
            #     process("chat", tools, qa)
            #     response = agent_chain.run(query)
            if memory not in st.session_state:
                st.session_state.memory = ConversationBufferWindowMemory(k=15)
            # response=llm_chain.memory.chat_memory.add_user_message(prompt)
            # st.write(memory.buffer)
            # #f"Echo: {prompt}" get_completion(template_string) #
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
