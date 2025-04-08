# NeuralDoc :scroll:  
**Intelligent Document Processing with Semantic Search & Dynamic PDF Generation**
A production-ready pipeline to transform static PDFs into an interactive knowledge base with AI-powered querying and LaTeX-based PDF regeneration.

---

## Key Features  
- **Multi-Stage PDF Parsing**  
  Extract structured text/images using PyMuPDF with layout preservation.
  
- **Vector-Powered Knowledge Base**  
  Store semantic embeddings in Milvus for lightning-fast similarity search.  

- **LLM-Augmented Responses**  
  Generate context-aware answers using Google Gemini and prompt engineering.  

- **LaTeX Publishing Pipeline**  
  Auto-convert Markdown to publication-ready PDFs with custom templates.  

- **Enterprise-Grade Containerization**  
  Fully Dockerized services with isolated environments.

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
