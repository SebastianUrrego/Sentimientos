import pypdf

reader = pypdf.PdfReader("Laboratorio 2 - Análisis de Sentimientos.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

with open("pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("Text extracted successfully.")
