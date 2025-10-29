import re
import argparse
import pdfplumber
import spacy
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Optional LLM support (attempt import; if unavailable we'll skip LLM)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

PAN_REGEX = re.compile(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b', re.IGNORECASE)

def load_pdf_text(path: Path):
    # returns list of strings (one per line-like chunk)
    chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # Extract text, split into lines to preserve proximity
            text = page.extract_text() or ""
            # Some PDFs have long lines; split on newline and on sentences
            for line in text.split('\n'):
                stripped = line.strip()
                if stripped:
                    chunks.append(stripped)
    return chunks

def extract_pans_from_text(text):
    return [m.group(1).upper() for m in PAN_REGEX.finditer(text)]

def run_spacy_ner(nlp, texts):
    # returns list of dicts with found persons and orgs per chunk
    results = []
    for t in texts:
        doc = nlp(t)
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        results.append({"text": t, "persons": persons, "orgs": orgs})
    return results

def build_candidate_relations(chunks_with_ner):
    # Heuristic: for each PAN found in chunk, pair with nearest PERSON in same chunk;
    # if no PERSON, pair with ORG if that makes sense (but PANs usually map to people).
    rows = []
    for c in chunks_with_ner:
        text = c["text"]
        pans = extract_pans_from_text(text)
        if not pans:
            continue
        for pan in pans:
            # prefer persons
            person = c["persons"][0] if c["persons"] else None
            org = c["orgs"][0] if c["orgs"] else None
            if person:
                rows.append({
                    "PAN": pan,
                    "Entity1_Type": "PAN",
                    "Entity1": pan,
                    "Relation": "PAN_Of",
                    "Entity2_Type": "Person",
                    "Entity2": person,
                    "Context": text
                })
            elif org:
                rows.append({
                    "PAN": pan,
                    "Entity1_Type": "PAN",
                    "Entity1": pan,
                    "Relation": "PAN_Of",
                    "Entity2_Type": "Organisation",
                    "Entity2": org,
                    "Context": text
                })
            else:
                # no entity detected: create candidate row with empty entity
                rows.append({
                    "PAN": pan,
                    "Entity1_Type": "PAN",
                    "Entity1": pan,
                    "Relation": "PAN_Of",
                    "Entity2_Type": "",
                    "Entity2": "",
                    "Context": text
                })
    return rows

# Optional: use an LLM to validate linkings. This is only invoked if user configures model.
def llm_validate_relations(rows, model_name="mistralai/mistral-7b-instruct", device_map="auto", use_hf_inference_api=False, hf_token=None):
    """
    Attempt to call a model to confirm/choose best matching person/org for each PAN based on context.
    This step is optional and will try to initialize a HF transformers pipeline.
    If unavailable, return rows unchanged.
    """
    if not LLM_AVAILABLE:
        print("Transformers not available locally; skipping LLM validation.")
        return rows

    # Simple prompt template: ask whether PAN maps to which person in context
    # For local models you may need to tune generation params.
    try:
        # For heavy models ensure you have proper environment setup.
        pipe = pipeline("text-generation", model=model_name, device_map=device_map, torch_dtype="auto")
    except Exception as e:
        print("Could not initialize model pipeline:", e)
        return rows

    validated = []
    for r in rows:
        context = r["Context"]
        pan = r["PAN"]
        prompt = (
            f"Context: {context}\n\n"
            f"Question: Which PERSON or ORGANISATION does the PAN {pan} belong to? "
            "Answer with a single line in the format: ENTITY_TYPE: ENTITY_NAME. "
            "If you are not sure, answer: UNKNOWN."
        )
        try:
            out = pipe(prompt, max_new_tokens=80, do_sample=False)[0]["generated_text"]
            # naive parsing:
            # look for lines like "PERSON: John Doe" or "ORGANISATION: ACME Ltd"
            m_person = re.search(r'(PERSON|Person|person)\s*[:\-]\s*(.+)', out)
            m_org = re.search(r'(ORGANISATION|Organization|ORG|Org|org)\s*[:\-]\s*(.+)', out)
            chosen = None
            if m_person:
                chosen = ("Person", m_person.group(2).strip().splitlines()[0])
            elif m_org:
                chosen = ("Organisation", m_org.group(2).strip().splitlines()[0])
            else:
                if "UNKNOWN" in out.upper():
                    chosen = None
                else:
                    # fallback: try to extract proper name-like token
                    lines = out.splitlines()
                    first = lines[0].strip()
                    if ":" in first:
                        _, val = first.split(":",1)
                        chosen = ("Person", val.strip())
            if chosen:
                etype, name = chosen
                r["Entity2_Type"] = etype
                r["Entity2"] = name
            # else keep original candidate (possibly empty)
        except Exception as e:
            print("LLM call failed for PAN", pan, ":", e)
        validated.append(r)
    return validated

def deduplicate_rows(rows):
    uniq = {}
    for r in rows:
        key = (r["PAN"], r.get("Entity2", ""), r.get("Entity2_Type",""))
        if key not in uniq:
            uniq[key] = r
        else:
            # prefer rows with non-empty context or entity
            if uniq[key]["Entity2"] == "" and r["Entity2"] != "":
                uniq[key] = r
    return list(uniq.values())

def main(pdf_path, out_csv, use_llm=False, llm_model=None):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("Loading spaCy model (this may take a few seconds)...")
    nlp = spacy.load("en_core_web_sm")

    print("Extracting text from PDF...")
    chunks = load_pdf_text(pdf_path)
    print(f"Extracted {len(chunks)} text chunks (lines) from PDF.")

    print("Running NER...")
    chunks_with_ner = run_spacy_ner(nlp, chunks)

    print("Building candidate relations using heuristics...")
    rows = build_candidate_relations(chunks_with_ner)
    print(f"Found {len(rows)} candidate PAN entries.")

    if use_llm and llm_model:
        print("Running LLM validation (may be slow) using model:", llm_model)
        rows = llm_validate_relations(rows, model_name=llm_model)
    else:
        print("LLM validation not requested or not configured; using heuristics only.")

    rows = deduplicate_rows(rows)
    df = pd.DataFrame(rows, columns=["PAN","Entity1_Type","Entity1","Relation","Entity2_Type","Entity2","Context"])
    df.to_csv(out_csv, index=False)
    print("Wrote results to:", out_csv)
    print(df.head(20).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PAN and PAN_Of relations from PDF")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out", default="extraction_result.csv", help="Output CSV path")
    parser.add_argument("--use-llm", action="store_true", help="Attempt to use LLM to validate relations (optional)")
    parser.add_argument("--llm-model", default="mistralai/mistral-7b-instruct", help="HF model name (if using LLM)")
    args = parser.parse_args()
    main(args.pdf, args.out, use_llm=args.use_llm, llm_model=args.llm_model)
