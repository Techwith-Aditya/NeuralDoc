# NeuralDoc :scroll:  
NeuralDoc is a robust application that transforms collections of PDF documents into a structured review paper with minimal effort. Here’s how it works:

- Multi-PDF Ingestion: Users upload multiple PDFs as raw inputs.

- Keyword-Based Similarity Search: A targeted keyword query drives a semantic similarity search, extracting the most relevant content from the dataset.

- Dynamic LaTeX Generation: The extracted content is automatically converted into structured LaTeX code.

- Review Paper PDF Output: Finally, the LaTeX code is compiled into a professionally formatted review paper.

This modular and extensible architecture is tailored for developers and researchers looking to automate document processing, streamline literature reviews, and generate polished academic outputs with ease.

---

## Key Features  
- **Multi-Stage PDF Parsing**  
  Extract structured text/images using PyMuPDF with layout preservation.
  
- **Keyword‑Driven Vector Search**  
  A user‑supplied keyword triggers a semantic similarity search to pinpoint the most relevant passages.  

- **Automated LaTeX Code Generation**  
  Transform retrieved content into fully‑formed LaTeX snippets—sections, citations, figures—using a template engine.

- **Review Paper PDF Assembly**  
  Compile the generated LaTeX into a single, publication‑ready review paper PDF with custom styling and bibliographic formatting. 

- **Enterprise-Grade Containerization**  
  Package each service (parsing, search, templating, compilation) into isolated Docker containers for easy deployment and scaling.

---

## Getting Started

### Prerequisites
- Docker Engine 24.0+ & Docker Compose v2.23+  
- Python 3.9+ (for local development)  
- API keys for Gemini (add to `.env`)

### Installation
```bash
git clone https://github.com/Techwith-Aditya/NeuralDoc.git
cd NeuralDoc
```
---

## Containerized Deployment

1. Start Milvus Vector Database
   ```bash
    docker compose up -d  
    ```
2. Build NeuralDoc Service
   ```bash
    docker build -t steamlit-app .  
    ```
3. Run the Application
   ```bash
    docker run -p 8501:8501 streamlit_app
    ```
4. Access the service at http://localhost:8501.

---

## Project Structure
```
NeuralDoc/  
├── data/                 # Raw PDF inputs
├── extracted/            # Parsed JSON/Markdown chunks
├── md_output/            # Processed Markdown
├── latex-output/         # LaTeX files (pre-compilation)
├── output/               # Final PDF outputs
│
├── parse.py              # PDF → Markdown parser (PyMuPDF)
├── retrieval.py          # Milvus vector search logic  
├── ToLatex.py            # LaTeX templating engine  
├── llm_prompt.py         # Gemini API integration  
├── docker-compose.yml    # Milvus orchestration  
└── Dockerfile            # Streamlit service container
```
---

## Contributing
- Fork the repository
- Create a feature branch (git checkout -b feature/your-feature)
- Submit a Pull Request targeting the main branch
