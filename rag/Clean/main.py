import os
import torch
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

"""# **2. CONFIGURATION**"""

LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# This determines the embedding model (for turning text into vectors)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Your PDF file (can be anything: resume, notes, textbook)
Txt_PATH = "/content/bio.txt"

# Directory to store the FAISS vector index
VECTOR_DB_DIR = "./faiss_index"

"""# **3. LOAD DOCUMENT (PDF)**"""

# Load PDF using LangChain's loader
loader = TextLoader(Txt_PATH)
pages = loader.load()  # Loads all pages as documents
print(f"Loaded {len(pages)} pages from PDF.")

"""# **4. SPLIT TEXT INTO CHUNKS**"""

# Breaks text into chunks to fit within LLM context window
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Number of characters in each chunk
    chunk_overlap=50       # Slight overlap helps retain context
)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks.")

"""# **5. EMBEDDINGS**"""

# Use sentence-transformers to turn chunks into vectors
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

"""# **6. CREATE VECTOR DB**"""

# Store and search document chunks using FAISS
db = FAISS.from_documents(chunks, embedding_model)

# Save the index for reuse (optional)
db.save_local(VECTOR_DB_DIR)
print(f"Vector DB created and saved at: {VECTOR_DB_DIR}")

"""# **7. LOAD LLM PIPELINE**"""

# Load tokenizer + model for causal generation
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)

# Create HuggingFace generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.7,
    device=device,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id  # add this line if using llm like phi2 etc
)

# Wrap the pipeline with LangChain's LLM interface
llm = HuggingFacePipeline(pipeline=llm_pipeline)

"""# **8. BUILD RAG CHAIN**"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant. Use only the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
)


# Combine vector retriever + LLM to form a RAG chatbot
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

"""# **9. ASK QUESTIONS!**"""

print("\nAsk questions from your document (type 'exit' to quit):")
while True:
    query = input("\nYou: ")
    if query.lower() == "exit":
        break

    result = qa_chain({"query": query})
    print("\nBot Answer:\n", result["result"])

    # Optional: Show the source document(s) used
    # print("\n Used Sources:")
    # for doc in result["source_documents"]:
    #     print("- ", doc.page_content[:200], "...")  # Truncated for readability
