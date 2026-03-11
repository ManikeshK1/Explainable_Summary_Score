import pandas as pd
import docx

def read_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

print("--- DOCX CONTENT ---")
text = read_docx("C:/Users/deii/Desktop/cloud/Agentic AI Class - 2025_08_07 18_10 GMT+05_30 - Transcript.docx")
print(text[:1000]) # Print first 1000 chars

print("\n--- XLSX CONTENT ---")
df = pd.read_excel("C:/Users/deii/Desktop/cloud/Summary.xlsx")
print(df.head())
