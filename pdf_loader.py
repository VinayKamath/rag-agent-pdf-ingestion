from pathlib import Path
import pymupdf
from langchain_core.documents import Document

class PDFLoader:
    def __init__(self, folder: str):
        self.folder = Path(folder)
        self.documents = []

    def load_pdfs(self):
        for pdf_path in self.folder.glob("*.pdf"):
            pdf = pymupdf.open(pdf_path)

            for page_num, page in enumerate(pdf):
                self.documents.append(
                    Document(
                        page_content=page.get_text(),
                        metadata={
                            "page": page_num + 1,
                            "total_pages": len(pdf),
                            "source": pdf_path.name,
                            "author": pdf.metadata.get("author"),
                            "title": pdf.metadata.get("title")                           
                        }
                    )
                )

            pdf.close()

        return self.documents

loader = PDFLoader("data")
documents = loader.load_pdfs()

print(f"Loaded {len(documents)} documents")
print(documents[0].metadata)
