import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub

# Load PDFs
all_docs = []
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"data/{file}")
        all_docs.extend(loader.load())

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.split_documents(all_docs)

# Embeddings (FREE)
embeddings = HuggingFaceEmbeddings()

# Vector DB
db = FAISS.from_documents(docs, embeddings)

# LLM (FREE API)
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

def get_answer(query):
    results = db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in results])
    sources = "\n".join([doc.metadata.get("source", "") for doc in results])

    prompt = f"""
    Answer clearly in the SAME language as the question.

    Context:
    {context}

    Question: {query}

    Give a short and accurate answer.
    """

    answer = llm.invoke(prompt)

    return answer + "\n\nSources:\n" + sources
