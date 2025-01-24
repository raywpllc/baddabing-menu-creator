from pdf_processor import PDFProcessor

if __name__ == "__main__":
    print("Starting PDF processing and embedding creation...")
    processor = PDFProcessor()
    processor.process_all_pdfs()
    print("Finished processing PDFs and creating embeddings!") 