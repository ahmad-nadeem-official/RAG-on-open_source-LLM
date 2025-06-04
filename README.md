RAG on Open-Source LLMs 🌟
==========================

Welcome to **RAG on Open-Source LLMs**, a cutting-edge project that brings Retrieval-Augmented Generation (RAG) to life using lightweight, open-source large language models (LLMs)! This repository demonstrates how to build a powerful, efficient, and scalable RAG pipeline to answer questions from documents with precision and speed. Whether you're a researcher, developer, or AI enthusiast, this project is your gateway to exploring the future of AI-driven question answering! 💡

🔗 **Repository Name**: RAG on Open-Source LLMs  
📅 **Last Updated**: June 2025  
👨‍💻 **Author**: Muhammad Ahmad Nadeem

* * *

⚠️ **Warning!** ⚠️

Please note that for the best performance, you should run this code on your system only if you have a GPU available on your machine other wise your system can be crash. If you don’t have a GPU, no worries! 🌐 I recommend running this code on **Google Colab**, where you can access free GPU resources effortlessly. Check out the Colab notebook below to get started:

🛡️ **[Google Colab](https://colab.research.google.com/drive/117VgcNYwceZWqy-SD7AalsnAkLLJAPhG#scrollTo=getMm5qOtO-q)**

![Generated Image](https://img.shields.io/badge/Google%20Colab-Run%20with%20GPU-blue?logo=googlecolab)

🌟 What is This Project About?
------------------------------

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using open-source LLMs, enabling users to query documents (like PDFs or text files) and get accurate, context-aware answers. By combining **vector search** with **LLM generation**, this pipeline retrieves relevant document chunks and generates precise responses—perfect for applications like research assistants, chatbots, or knowledge management systems! 📚

The core idea is to make RAG accessible and efficient by leveraging lightweight models like **TinyLlama** and open-source tools, ensuring you can run this pipeline even on modest hardware (yes, even on a CPU if needed)! 💻

* * *

🚀 Key Features
---------------

*   **Document Ingestion** 📜: Load and process text or PDF documents effortlessly.
*   **Text Chunking** ✂️: Smartly split documents into manageable chunks for better context retention.
*   **Vector Search** 🔍: Use FAISS to create a vector database for fast and accurate retrieval.
*   **Open-Source LLM** 🤖: Leverage TinyLlama (1.1B parameters) for efficient text generation.
*   **Custom Prompts** ✍️: Tailor the LLM's behavior with custom prompt templates.
*   **Scalable Pipeline** ⚙️: Modular code structure for easy experimentation and scaling.
*   **Hardware Flexibility** 🖥️: Run on GPU (CUDA) or CPU, depending on your setup.

* * *

🛠️ Tech Stack
--------------

Here’s the powerhouse of tools and libraries that make this project tick!

🛡️ Python

![Generated Image](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)

  
🛡️ PyTorch

![Generated Image](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)

  
🛡️ LangChain

![Generated Image](https://img.shields.io/badge/LangChain-0.1%2B-green?logo=langchain)

  
🛡️ HuggingFace Transformers

![Generated Image](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)

  
🛡️ FAISS

![Generated Image](https://img.shields.io/badge/FAISS-Vector%20Search-blue)

  
🛡️ Sentence-Transformers

![Generated Image](https://img.shields.io/badge/Sentence--Transformers-Embeddings-purple)

* * *

📂 Project Structure
--------------------

Here’s a quick look at how the repository is organized:

*   **rag/**: Core directory for RAG pipeline scripts.
    *   **clean**: Folder where I saved main file.
          *  **main.py**: main file of code
    *   **RAW**: **Jupyter notebook** and **rag.py** for interactive experimentation and step-by-step walkthrough.
    *   **SRC/**: Source directory for additional scripts or utilities.
      *   **bio.txt**: Sample text file used for testing the RAG pipeline.
*   **.gitignore**: Gitignore file to keep the repo clean.
*   **README.md**: You’re reading it! 😄
*   **requirements.txt**: List of dependencies to set up the project.

* * *

🖥️ How to Get Started
----------------------

Follow these steps to set up and run the project on your machine! 🚀

### Prerequisites

*   Python 3.8 or higher 🐍
*   A CUDA-enabled GPU (optional but recommended for faster performance) 💻
*   Git installed to clone the repository 📦

### Step 1: Clone the Repository

git clone [https://github.com/\[your-username\]/RAG-on-open-source-LLMs.git](https://github.com/%5Byour-username%5D/RAG-on-open-source-LLMs.git)  
cd RAG-on-open-source-LLMs

### Step 2: Install Dependencies

pip install -r requirements.txt

### Step 3: Prepare Your Document

Place your document (e.g., a PDF or text file) in the **RAW/** directory. By default, the pipeline uses **bio.txt** as a sample file.

### Step 4: Run the Pipeline

python rag/main.py

This will:

1.  Load your document.
2.  Split it into chunks.
3.  Create a FAISS vector database.
4.  Load the TinyLlama model.
5.  Start an interactive Q&A session where you can ask questions!

### Step 5: Ask Questions!

Once the pipeline is running, type your question (e.g., "What is the main topic of the document?") and get a precise answer. Type **exit** to quit.

* * *

📊 How It Works
---------------

Here’s a high-level overview of the RAG pipeline in this project:

1.  **Load Document** 📜: The pipeline uses LangChain’s TextLoader to load a text file (e.g., bio.txt).
2.  **Split into Chunks** ✂️: The document is split into smaller chunks (500 characters each) with a slight overlap to retain context.
3.  **Create Embeddings** 🔢: Chunks are converted into vectors using **sentence-transformers/all-MiniLM-L6-v2**.
4.  **Build Vector DB** 🗄️: FAISS creates a searchable vector database from the embeddings.
5.  **Load LLM** 🤖: TinyLlama (1.1B) is loaded via HuggingFace Transformers for text generation.
6.  **Set Up RAG Chain** ⚙️: LangChain’s RetrievalQA combines the vector retriever and LLM with a custom prompt.
7.  **Query and Answer** 💬: Ask a question, and the pipeline retrieves relevant chunks and generates an answer!

* * *

🧠 Why This Project Stands Out
------------------------------

*   **Efficiency First** ⚡: Uses lightweight models like TinyLlama, making it accessible for users without high-end GPUs.
*   **Modular Design** 🧩: Easily swap out components (e.g., change the LLM or embedding model) for experimentation.
*   **Production-Ready** 🚀: Clean, well-documented code that’s ready for deployment in real-world applications.
*   **Open-Source Focus** 🌍: Built entirely with open-source tools, promoting accessibility and collaboration.

* * *

🤝 Contributing
---------------

We’d love for you to contribute to this project! Here’s how:

1.  Fork the repository 🍴
2.  Create a new branch (git checkout -b feature/awesome-feature)
3.  Commit your changes (git commit -m "Add awesome feature")
4.  Push to the branch (git push origin feature/awesome-feature)
5.  Open a Pull Request 📬

Feel free to open issues for bugs, feature requests, or suggestions! Let’s make this project even better together. 🤗

* * *

📜 License
----------

This project is licensed under the MIT License. See the LICENSE file for details.

* * *

🙌 Acknowledgments
------------------

*   **HuggingFace** for their amazing open-source LLMs and Transformers library.
*   **LangChain** for simplifying the RAG pipeline creation.
*   **FAISS** for efficient vector search.
*   **Sentence-Transformers** for high-quality embeddings.

* * *

📧 Contact
----------

Have questions or want to collaborate? Reach out to me at \[[your-email@example.com](mailto:your-email@example.com)\] or connect with me on \[LinkedIn/Twitter\]! Let’s build something amazing together. 🚀

* * *

🎉 **Thank you for checking out RAG on Open-Source LLMs!** Let’s revolutionize the way we interact with documents using AI. Happy coding! 💻

