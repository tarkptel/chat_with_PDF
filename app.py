import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline  # 👈 Import LangChain's transformer wrapper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import torch
from dotenv import load_dotenv
#from sentence_transformers import SentenceTransformer

load_dotenv()

hf_token = os.getenv("HF_TOKEN")  # Must match the secret name

st.set_page_config(page_title="Chat with PDF (RAG)", layout="wide")
st.title("📄💬 Chat with PDF (RAG using LangChain + Transformers)")

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    with st.spinner("Reading and processing PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())

        loader = PyPDFLoader('temp.pdf')
        docs = loader.load()
        texts = [doc.page_content for doc in docs]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents(texts)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embedding_model)

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        # ✅ Load Transformers model and tokenizer
       
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

        # Wrap the transformers model in LangChain's HuggingFacePipeline
        pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.6,
        repetition_penalty=1.1,
            )
        
        llm = HuggingFacePipeline(pipeline=pipe)

        # 🧠 Prompt template
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an intelligent assistant that answers questions based on the provided context.
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )

        # 🧱 LangChain runnable pipeline
        parser = StrOutputParser()

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        main_chain = parallel_chain | prompt | llm | parser

    question = st.text_input("Ask a question about the PDF...")

    if question:
        with st.spinner("Thinking..."):
            result = main_chain.invoke(question)
        st.markdown("### 💬 Answer")
        st.write(result)
