import streamlit as st
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

st.title("🏄‍♂️ SiFly Chatbot")


@st.cache_resource
def get_loader():
    return PyPDFLoader("data/SiFly_User_Guide.pdf")


@st.cache_resource
def get_vectorstore(_loader):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    embeddings = OpenAIEmbeddings()

    docs = _loader.load()
    documents = text_splitter.split_documents(docs)
    return Chroma.from_documents(documents, embeddings, collection_name="sifly")


@st.cache_data
def get_llm():
    """Chat Completition LLM"""
    return ChatOpenAI(
        temperature=0.0,
    )


@st.cache_resource
def get_sifly_tool(_llm, _retriever):
    # retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=_llm,
        retriever=_retriever,
        chain_type="stuff",
    )

    return Tool(
        name="SiFly Q&A System",
        func=qa.run,
        description="""useful for when you need to answer
        questions about SiFly (an eFoil brand).
        Input should be a fully formed question.""",
    )


@st.cache_resource
def get_agent(_llm, _tools):
    # conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True,
    )

    return initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=_tools,
        llm=_llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        memory=conversational_memory,
    )


def generate_response(input_text: str) -> str:
    llm = get_llm()
    retriever = get_vectorstore(get_loader()).as_retriever()
    tools = [get_sifly_tool(llm, retriever)]
    agent = get_agent(llm, tools)

    return agent(input_text)["output"]


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = generate_response(prompt)
    msg = {"role": "assistant", "content": response}
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg["content"])
