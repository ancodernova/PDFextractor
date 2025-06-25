import os
import uuid
from sqlalchemy import create_engine, Column, String, Text, Table, MetaData
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
# --- SQLite Setup ---
engine = create_engine("sqlite:///saved_answers.db")
metadata = MetaData()

saved_table = Table(
    "saved_answers", metadata,
    Column("id", String, primary_key=True),
    Column("question", Text),
    Column("answer", Text),
    Column("source", Text)
)
metadata.create_all(engine)

# --- Embeddings ---
def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",  # Or "text-embedding-ada-002"
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# --- PDF Vectorization ---
def load_and_vectorize(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    embeddings = get_embeddings()
    vector_db = FAISS.from_documents(docs, embeddings)

    vector_db.save_local("faiss_index")  # Persist FAISS
    return vector_db

# --- Reload Existing FAISS ---
def load_faiss_if_exists():
    if os.path.exists("faiss_index/index.faiss"):
        embeddings = get_embeddings()
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None

# --- QA Chain ---
def create_qa_chain(vector_db):
    llm = ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt_template = """
    Use the context below to answer the question. Include the page number and quoted sentence if available.
    If the answer is not known, say "I don't know".

    Question: {question}
    =========
    {context}
    =========
    Helpful answer with reference:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain

# --- Save to DB ---
def save_to_db(question, answer, source):
    with engine.connect() as conn:
        conn.execute(saved_table.insert().values(
            id=str(uuid.uuid4()),
            question=question,
            answer=answer,
            source=source
        ))
