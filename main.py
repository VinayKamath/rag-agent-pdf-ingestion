from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from pdf_loader import PDFLoader
from document_splitter import DocumentSplitter
from embeddings import Embeddings


DATA_FOLDER = "data"
TOP_K = 3
MODEL_NAME = "gpt-5.5"


def build_context(documents: list[Document]) -> str:
    """Format retrieved documents into a readable context string."""
    sections = []

    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        content = doc.page_content.strip()

        sections.append(
            f"[{i}] Source: {source} | Page: {page}\n{content}"
        )

    return "\n\n".join(sections)


def retrieve_relevant_chunks(vectorstore, question: str, k: int = TOP_K) -> list[Document]:
    """Retrieve the most relevant chunks for a question."""
    return vectorstore.similarity_search(question, k=k)


def answer_question(vectorstore, question: str) -> str:
    """Retrieve context and generate an answer using the LLM."""
    results = retrieve_relevant_chunks(vectorstore, question)

    if not results:
        return "No relevant documents were found."

    context = build_context(results)

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
    )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Answer the user's question using only "
                "the provided context. If the context does not contain enough "
                "information, say that clearly. When relevant, refer to the source "
                "and page numbers from the context."
            )
        ),
        HumanMessage(
            content=f"Question:\n{question}\n\nContext:\n{context}"
        ),
    ]

    response = llm.invoke(messages)
    return response.content


def main() -> None:
    loader = PDFLoader(DATA_FOLDER)
    docs = loader.load_pdfs()

    if not docs:
        print("No PDF documents found.")
        return

    splitter = DocumentSplitter()
    chunks = splitter.split_documents(docs)

    if not chunks:
        print("No chunks were created from the documents.")
        return

    embeddings = Embeddings(chunks)
    vectorstore = embeddings.get_vectorstore()

    question = "Can you summarize the CHASE consumer policy?"
    answer = answer_question(vectorstore, question)

    print(answer)


if __name__ == "__main__":
    main()