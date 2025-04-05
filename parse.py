from dotenv import load_dotenv
import os 
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader 

load_dotenv()

api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# Initialized with premium mode for better table/chart extraction
parser = LlamaParse(
    result_type="markdown",
    premium_mode=True,
    api_key=api_key
)

pdf_file_path = "cnn1.pdf"
output_md_path = "md_output/output1.md"

os.makedirs(os.path.dirname(output_md_path), exist_ok=True)

try:
    # Set up custom PDF parser configuration
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[pdf_file_path], file_extractor=file_extractor).load_data()

    if not documents:
        raise ValueError("No data was parsed from the provided PDF.")

    markdown_content = "\n\n".join([doc.text for doc in documents])

    print(markdown_content)

    with open(output_md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)

    print(f"Markdown saved to: {output_md_path}")

except Exception as e:
    # Wrap exception to provide context about the failure
    raise ValueError(f"Error parsing PDF: {e}")
