import os
import openai
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile
import time

# ✅ Set OpenAI API Key Directly
openai.api_key = "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"  # 🔹 Replace with your actual OpenAI API key

# ✅ Fine-Tuned Model (Replace with your fine-tune ID)
FINE_TUNED_MODEL = "davinci:ft-YOUR-FINE-TUNE-ID"

# ✅ Persona - Strictly Fact-Based
persona = '''
You are an expert research assistant specializing in **financial reporting and corporate transparency**.
Your responses must be:

1️⃣ **Strictly Fact-Based & Source-Driven**  
   - Only provide answers based on verified SEC 10-K reports of Amazon, Alphabet, and Microsoft.  
   - Cite relevant sections **explicitly** using `[i]` notation.  
   - If a statement lacks clear evidence, explicitly state the gap.  

2️⃣ **Deep Analysis Without Hallucination**  
   - Do **not fabricate numbers, trends, or insights** not explicitly mentioned in the reports.  
   - If a user asks about missing details, suggest external reports **but do not generate false claims**.  

3️⃣ **Comprehensive, Clear, & Detailed Responses**  
   - Instead of **summaries**, provide **detailed explanations** backed by evidence.  
   - **Compare & contrast** different sections of the document if relevant.  

4️⃣ **Robust Citation & Verification**  
   - Clearly mark which parts of the response are directly from the document `[i]`.  
   - If multiple sources are used, cite all relevant ones `[i, j, k]`.  

5️⃣ **Transparent About Limitations**  
   - If the document **does not contain an answer**, state it outright.  
   - Do **not assume missing information**—instead, guide the user to relevant sources.  

Your goal is to provide **precise, evidence-backed answers** with a focus on **accuracy and document fidelity**.
'''

# ✅ Prompt Template
template = """
{persona}

Chat History:
<history>
{chat_history}
</history>

Question: {user_input}
"""

# ✅ Set up Streamlit UI
st.set_page_config(page_title="Chat with 10-K Reports (Amazon, Alphabet, Microsoft)")
st.title("📄💬 Chat with 10-K Reports (Amazon, Alphabet, Microsoft)")

# ✅ File Uploader (Multiple PDFs)
uploaded_files = st.file_uploader("Upload 10-K PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    documents = []
    
    if "vector_store" not in st.session_state:
        with st.spinner("Processing PDFs..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())

                    # ✅ Load PDFs
                    loader = PyPDFLoader(temp_file_path)
                    documents.extend(loader.load())

                # ✅ Chunking for better retrieval
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)

                # ✅ Store embeddings in FAISS
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    openai_api_key="your-openai-api-key-here"  # 🔹 API Key is now passed directly
                )
                st.session_state.vector_store = FAISS.from_documents(docs, embeddings)

        st.success("✅ PDFs processed! You can now start asking questions.")

    # ✅ Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ✅ Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ✅ User Input
    user_input = st.chat_input("Ask a question about the 10-K reports...")

    if user_input:
        # ✅ Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)

        # ✅ Configure improved retrieval settings
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",  
            search_kwargs={
                "k": 12,  # Retrieve more chunks for deeper analysis
                "fetch_k": 30,  
                "lambda_mult": 0.6,  
                "score_threshold": 0.3  
            }
        )

        # ✅ Get chat history
        chat_history = ""
        if len(st.session_state.messages) > 1:
            for i, msg in enumerate(st.session_state.messages[:-1]):  
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_history += f"{role}: {msg['content']}\n\n"
        
        # ✅ Fine-Tuned LLM Query
        try:
            response = openai.ChatCompletion.create(
                model=FINE_TUNED_MODEL,
                messages=[
                    {"role": "system", "content": persona},
                    {"role": "user", "content": user_input}
                ]
            )

            # ✅ Display Assistant Response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = response["choices"][0]["message"]["content"]
                message_placeholder.markdown(full_response)

            # ✅ Store assistant response in chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"⚠️ OpenAI API Error: {str(e)}")

else:
    st.info("Please upload 10-K reports to begin.")
