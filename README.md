# 🧾 PAN Entity & Relation Extraction

This project extracts **PAN numbers**, associated **Names** and **Organisations** from PDF documents and identifies the **PAN_Of** relationship using Python.

---

## 🚀 Project Overview

The goal is to automate the extraction of entities and relations from text-heavy documents (like Aadhaar or tax forms).  
It uses open-source NLP tools and regular expressions to maximize correct extractions with minimal false positives.

**Extracted Entities**
- **Organisation**
- **Name (Person)**
- **PAN**

**Relation**
- `PAN_Of` — links a PAN to a person or organisation

## 🧰 Technologies Used

- **Python 3.10+**
- **pdfplumber** – extract text from PDF files  
- **spaCy** – detect person and organisation entities  
- **pandas** – create and export structured CSV results  
- **regex (re)** – detect valid PAN patterns  
- *(Optional)* **LLM validation** – open-source models like *Mistral 7B* can validate extracted relations  

## 📂 Project Files

| File | Description |
|------|--------------|
| `extract_pan_relations.py` | Main script for entity and relation extraction |
| `sample_filled_aadhaar_like.pdf` | Sample input PDF used for testing |
| `result.csv` | Extracted entities and relations |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
