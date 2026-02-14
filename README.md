# 🏥 LLM-Powered Medical Report Summarization System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/)

A retrieval-grounded medical report analysis system that combines LoRA fine-tuned TinyLlama, BioBERT embeddings, and Qdrant vector search to generate clinically-aware summaries and structured medical insights from MIMIC-IV microbiology reports.

## 🎯 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for medical report analysis using:

🔹 TinyLlama-1.1B (LoRA fine-tuned)
🔹 BioBERT-based sentence embeddings
🔹 Qdrant vector database
🔹 Biomedical Named Entity Recognition (NER)
🔹 Entity-level evaluation (Precision, Recall, F1)

The system compares:

- Vanilla LLM generation (without retrieval)
- RAG-grounded generation (with retrieval)

and evaluates clinical correctness using automated entity extraction.

## ✨ Key Features

- 🤖 LoRA Fine-Tuned TinyLlama (1.1B)
- 🔍 Retrieval-Augmented Generation (RAG) using Qdrant
- 🧠 BioBERT-based Embeddings for semantic retrieval
- 🏥 Biomedical NER-based Evaluation
- 📊 Entity-Level Precision, Recall, F1
- ⚡ Efficient small-model deployment experimentation
- 🔬 Evaluation pipeline robust to token fragmentation & structured artifacts

## 🏗️ System Architecture
```
Medical Report Input
    ↓
BioBERT Embedding
    ↓
Qdrant Vector Search
    ↓
User Query
    ↓
Top-K Medical Report Retrieval
    ↓
TinyLlama (LoRA Fine-Tuned)
    ↓
Generated Medical Analysis
    ↓
Biomedical NER Extraction
    ↓
Entity-Level Evaluation (P/R/F1)

```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Base Model** | TinyLlama-1.1B |
| **Fine-tuning** | LoRA (Parameter-Efficient Fine-Tuning) |
| **RAG Framework** | LangChain |
| **Vector Database** | Qdrant |
| **Embeddings** | sentence-transformers |
| **Dataset** | MIMIC-IV Clinical Notes |
| **NER** | d4data/biomedical-ner-all |
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch, Hugging Face Transformers |

## 📋 Prerequisites
```bash
Python 3.8 or higher
CUDA-capable GPU (recommended for training)
8GB+ RAM
Access to MIMIC-IV dataset (requires PhysioNet credentialing)
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/abhisheknagare/LLM-Powered-Medical-Report-Summarization-System.git
cd LLM-Powered-Medical-Report-Summarization-System
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download MIMIC-IV Dataset
```bash
# 1. Complete PhysioNet credentialing: https://physionet.org/
# 2. Download MIMIC-IV dataset
# 3. Place in data/mimic-iv/ directory
```

## 💻 Usage

### Training the Model
```python
# Fine-tune TinyLlama with LoRA on MIMIC-IV data
python train.py \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset_path data/mimic-iv/notes \
    --output_dir models/tinyllama-medical \
    --lora_r 8 \
    --lora_alpha 16 \
    --num_epochs 3 \
    --batch_size 4
```

### Inference Example
```python
from medical_summarizer import MedicalReportSummarizer

# Initialize the summarizer
summarizer = MedicalReportSummarizer(
    model_path="models/tinyllama-medical",
    vector_db_path="data/vector_db"
)

# Sample medical report
report = """
Chief Complaint: Chest pain
History of Present Illness: 65-year-old male presents with substernal chest pain...
Physical Examination: BP 140/90, HR 88, RR 16, Temp 98.6°F
Assessment and Plan: Rule out acute coronary syndrome...
"""

# Generate summary
summary = summarizer.summarize(report)
print("Summary:", summary)

# Extract clinical findings
findings = summarizer.extract_findings(report)
print("Clinical Findings:", findings)
```

### RAG Pipeline Usage
```python
from rag_pipeline import RAGPipeline

# Initialize RAG system
rag = RAGPipeline(
    llm_model="models/tinyllama-medical",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    vector_db_type="faiss"
)

# Query with context retrieval
query = "What are the patient's vital signs and chief complaint?"
response = rag.query(query, report_text)
print(response)
```

## 📊 Performance Metrics

| Model Version | Precision | Recall | F1 Score
|--------|---------|---------|---------|
| Without Retrieval | Moderate | Moderate | Baseline |
| With Retrieval (RAG) | Higher | Higher | Improved |

- RAG grounding improves clinical entity recall and reduces hallucinated content compared to vanilla generation.

### Evaluation Results

- **Precision**: High accuracy in identifying diagnoses, medications, and vital signs
- **Recall**: Effectively captures key clinical information from unstructured text
- **F1-Score**: Balanced performance across different clinical finding categories

## 📁 Project Structure
```
LLM-Powered-Medical-Report-Summarization-System/
│
├── data/
│   ├── mimic-iv/              # MIMIC-IV dataset (not included)
│   ├── vector_db/             # Vector database storage
│   └── processed/             # Preprocessed data
│
├── models/
│   ├── tinyllama-medical/     # Fine-tuned model checkpoints
│   └── embeddings/            # Embedding models
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── train.py               # Model training script
│   ├── inference.py           # Inference utilities
│   ├── medical_summarizer.py  # Main summarizer class
│   ├── rag_pipeline.py        # RAG implementation
│   ├── data_processing.py     # Data preprocessing
│   └── evaluation.py          # Evaluation metrics
│
├── tests/
│   └── test_summarizer.py
│
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── .gitignore
├── LICENSE
└── README.md
```

## 🔬 Technical Details

### Model Architecture

**Base Model**: TinyLlama-1.1B-Chat-v1.0
- Decoder-only transformer architecture
- 1.1 billion parameters
- Optimized for efficient inference

**LoRA Configuration**:
```python
lora_config = {
    "r": 8,                    # Rank of update matrices
    "lora_alpha": 16,          # Scaling factor
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,
    "bias": "none"
}
```

### RAG Pipeline

**Components**:
1. **Document Chunking**: Splits reports into semantic chunks
2. **Embedding Generation**: Creates vector representations using sentence-transformers
3. **Vector Storage**: Qdrant for efficient similarity search
4. **Retrieval**: Top-k relevant chunks based on query
5. **Generation**: LLM generates response using retrieved context

**Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional embeddings
- Balanced performance and speed

## 🔐 Data Privacy & Compliance

⚠️ **IMPORTANT NOTICE**

This project uses the **MIMIC-IV dataset**, which contains de-identified patient data:

- ✅ All data is de-identified per HIPAA Safe Harbor standards
- ✅ PhysioNet credentialing required for dataset access
- ✅ No patient-identifiable information is processed or stored
- ⚠️ For production use, ensure compliance with local regulations (HIPAA, GDPR, etc.)
- ⚠️ Clinical validation required before deployment in healthcare settings

**PhysioNet Access**: https://physionet.org/content/mimiciv/

## 🎓 Key Learnings

This project demonstrates:
- **Parameter-efficient fine-tuning** using LoRA for domain adaptation
- **RAG architecture** for grounding LLM outputs in factual medical data
- **Vector databases** for efficient semantic search in clinical text
- **Clinical NLP** challenges and solutions
- **Production considerations** for healthcare AI systems

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- Tests pass for new features
- Documentation is updated

## 🐛 Known Issues

- Large reports (>2000 tokens) may require chunking for optimal performance
- GPU memory requirements: minimum 8GB VRAM for training
- RAG retrieval quality depends on embedding model choice

## 🙏 Acknowledgments

- **MIMIC-IV Dataset**: Johnson, A., Bulgarelli, L., Pollard, T., et al. (MIT LCP)
- **TinyLlama Team**: For the efficient base model
- **Hugging Face**: Transformers library and PEFT implementation
- **LangChain**: RAG framework and utilities
- **PhysioNet**: For providing access to clinical datasets

## 📚 References

1. Johnson, A. E. W. et al. (2023). MIMIC-IV (version 2.2). PhysioNet.
2. Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
3. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.

## 📧 Contact

**Abhishek Nagare**

- 📧 Email: abhisheknagare01@gmail.com
- 💼 LinkedIn: [linkedin.com/in/abhishekmnagare](https://www.linkedin.com/in/abhishekmnagare)
- 🐙 GitHub: [@abhisheknagare](https://github.com/abhisheknagare)

## 📖 Citation

If you use this project in your research or work, please cite:
```bibtex
@software{nagare2025medical_llm,
  author = {Nagare, Abhishek},
  title = {LLM-Powered Medical Report Summarization System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/abhisheknagare/LLM-Powered-Medical-Report-Summarization-System}
}
```

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=abhisheknagare/LLM-Powered-Medical-Report-Summarization-System&type=Date)](https://star-history.com/#abhisheknagare/LLM-Powered-Medical-Report-Summarization-System&Date)

---

<div align="center">

**Made with ❤️ for Healthcare AI**

If you find this project useful, please consider giving it a ⭐!

</div>

---

> **Disclaimer**: This is an academic/research project. Not intended for clinical use without proper validation, regulatory approval, and clinical oversight. Always consult healthcare professionals for medical decisions.
