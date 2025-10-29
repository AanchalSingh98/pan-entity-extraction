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

**Example:**
