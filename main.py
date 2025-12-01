# Declare imports
import gradio as gr
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile
from dotenv import load_dotenv


# Initialize clients (set your API keys as environment variables)
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY", "")
pinecone_key = os.getenv("PINECONE_API_KEY", "")

# Set Pinecone index name
PINECONE_INDEX_NAME = "rag-qa-index"


class RAGPipeline:
    """Class to handle RAG pipeline operations"""

    def __init__(self):
        """Initialize RAG pipeline components"""
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.pc = None
        self.index = None

    def initialize_pinecone(self, api_key):
        """Initialize Pinecone client and create/connect to index"""
        try:
            self.pc = Pinecone(api_key=api_key)

            # Check if index exists, if not create it
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if PINECONE_INDEX_NAME not in existing_indexes:
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=1536,  # OpenAI embeddings dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )

            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            return "✓ Pinecone initialized successfully"
        except Exception as e:
            return f"✗ Pinecone initialization failed: {str(e)}"

    def process_document(self, file, chunk_size, chunk_overlap):
        """Process uploaded document and store in Pinecone"""
        try:
            # Initialize APIs
            os.environ["OPENAI_API_KEY"] = openai_key
            pinecone_status = self.initialize_pinecone(pinecone_key)

            if "failed" in pinecone_status:
                return pinecone_status

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                tmp_file.write(file.read() if hasattr(
                    file, 'read') else open(file.name, 'rb').read())
                tmp_path = tmp_file.name

            # Load document based on file type
            if file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            elif file.name.endswith('.txt'):
                loader = TextLoader(tmp_path)
            else:
                return "✗ Unsupported file format. Please upload PDF or TXT file."

            documents = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)

            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

            # Create vector store
            self.vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                index_name=PINECONE_INDEX_NAME
            )

            # Initialize QA chain
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0,
                openai_api_key=openai_key
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )

            # Clean up temporary file
            os.unlink(tmp_path)

            return f"✓ Document processed successfully!\n- File: {file.name}\n- Chunks created: {len(chunks)}\n- Ready for questions!"

        except Exception as e:
            return f"✗ Error processing document: {str(e)}"

    def answer_question(self, question):
        """Answer question using RAG pipeline"""
        if not self.qa_chain:
            return "⚠ Please upload and process a document first!"

        if not question.strip():
            return "⚠ Please enter a question!"

        try:
            result = self.qa_chain.invoke({"query": question})

            answer = result['result']
            sources = result.get('source_documents', [])

            # Format response with sources
            response = f"**Answer:**\n{answer}\n\n"

            if sources:
                response += "**Sources:**\n"
                for i, doc in enumerate(sources[:3], 1):
                    content_preview = doc.page_content[:200] + "..." if len(
                        doc.page_content) > 200 else doc.page_content
                    response += f"\n{i}. {content_preview}\n"

            return response

        except Exception as e:
            return f"✗ Error answering question: {str(e)}"


# Initialize pipeline
pipeline = RAGPipeline()

# Create Gradio interface
with gr.Blocks(title="RAG Q&A Pipeline") as demo:
    gr.Markdown("# 📚 RAG Q&A Pipeline")
    gr.Markdown(
        "Upload a document (PDF or TXT) and ask questions about its content using AI-powered retrieval.")

    with gr.Row():
        with gr.Column(scale=1):

            gr.Markdown("### 📄 Document Upload")
            file_input = gr.File(file_types=[".pdf", ".txt"])

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                chunk_size = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=1000,
                    step=100,
                    label="Chunk Size"
                )
                chunk_overlap = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=200,
                    step=50,
                    label="Chunk Overlap"
                )

            process_btn = gr.Button("🚀 Process Document", variant="primary")
            status_output = gr.Textbox(
                label="Status",
                lines=5,
                interactive=False
            )

        with gr.Column(scale=1):
            gr.Markdown("### 💬 Ask Questions")
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask anything about the uploaded document...",
                lines=3
            )
            ask_btn = gr.Button("🔍 Get Answer", variant="primary")
            answer_output = gr.Markdown(label="Answer")

            gr.Markdown("### 📝 Example Questions")
            gr.Examples(
                examples=[
                    ["What is the main topic of this document?"],
                    ["Can you summarize the key points?"],
                    ["What are the main conclusions?"],
                ],
                inputs=question_input
            )

    # Event handlers
    process_btn.click(
        fn=pipeline.process_document,
        inputs=[file_input, chunk_size, chunk_overlap],
        outputs=status_output
    )

    ask_btn.click(
        fn=pipeline.answer_question,
        inputs=question_input,
        outputs=answer_output
    )

    question_input.submit(
        fn=pipeline.answer_question,
        inputs=question_input,
        outputs=answer_output
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
