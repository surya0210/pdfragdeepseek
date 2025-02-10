from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from pypdf import PdfReader
import gradio as gr
import asyncio
#import streamlit as st

#For reading through all the docs and inserting them into ChromaDB and creating a pipeline.
async def async_initialize_rag(pdf_files):
    try:
        documents = []
        for pdf_file in pdf_files:
            with open(pdf_file.name, "rb") as f:
                pdf = PdfReader(f)
                documents.extend([
                    Document(
                        page_content=page.extract_text(),
                        metadata={"source": pdf_file.name, "page": i+1}
                    )
                    for i, page in enumerate(pdf.pages)
                    if page.extract_text().strip()  # Skip empty pages
                ])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        #using deepseek
        embeddings = OllamaEmbeddings(
            model="deepseek-r1"
        
        )
        
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        llm = OllamaLLM(
            model="deepseek-r1",
            temperature=0.3,
            num_ctx=4096
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
    except Exception as e:
        raise RuntimeError(f"Initialization failed: {str(e)}")

async def async_ask_question(query, pipeline):
    try:
        response = await asyncio.to_thread(pipeline.invoke, {"query": query})
        answer = response["result"]
        sources = "\n".join(
            f"• {doc.metadata['source']} (page {doc.metadata.get('page', 'N/A')})"
            for doc in response["source_documents"]
        )
        return f"{answer}\n\nSources:\n{sources}"
    except Exception as e:
        return f"Error processing query: {str(e)}"

def create_interface():
    with gr.Blocks(title="Bujji") as interface:
        gr.Markdown("Bujji-PDF")
        
        with gr.Row():
            with gr.Column():
                file_upload = gr.File(
                    file_count="multiple",
                    label="Upload PDF Document",
                    type="filepath"
                )
                init_btn = gr.Button("Initialize System", variant="primary")
                status = gr.Textbox(label="System Status", interactive=False)
            
            with gr.Column():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask me anything",
                    lines=3
                )
                ask_btn = gr.Button("Ask Question", variant="secondary")
                response_output = gr.Textbox(
                    label="Answer",
                    interactive=False,
                    lines=6
                )
        
        # State management
        pipeline_state = gr.State()

        # Event handlers
        init_btn.click(
            fn=async_initialize_rag,
            inputs=file_upload,
            outputs=pipeline_state,
            api_name="init_rag"
        ).then(
            lambda: "System Ready - Lets go",
            outputs=status
        )

        ask_btn.click(
            fn=async_ask_question,
            inputs=[question_input, pipeline_state],
            outputs=response_output,
            api_name="ask_question"
        )

    return interface

if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None
    )
