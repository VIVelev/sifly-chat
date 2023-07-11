import streamlit as st
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

st.title("🏄‍♂️ SiFly Chatbot")


@st.cache_resource
def get_loader():
    return PyPDFLoader("data/SiFly_User_Guide.pdf")


@st.cache_resource
def get_sifly_db():
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    embeddings = OpenAIEmbeddings()

    docs = get_loader().load()
    sifly_texts = text_splitter.split_documents(docs)
    return Chroma.from_documents(sifly_texts, embeddings, collection_name="sifly")


def get_sifly_tool(llm):
    sifly = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=get_sifly_db().as_retriever()
    )

    return sifly


def generate_response(input_text):
    llm = OpenAI(temperature=0)

    tools = [
        Tool(
            name="SiFly QA System",
            func=get_sifly_tool(llm).run,
            description="useful for when you need to answer questions about SiFly (an eFoil brand). Input should be a fully formed question.",
        ),
    ]

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    st.info(agent.run(input_text))


with st.form("my_form"):
    text = st.text_area("Enter text:", "How to connect the remote to the board?")
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
