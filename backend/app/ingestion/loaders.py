from pathlib import Path

def load_text_document(file_path):

    text = file_path.read_text(encoding="utf-8")

    lines = text.splitlines()

    metadata = {}
    content_lines = []
    reading_metadata = True

    for line in lines:
        clean_line = line.strip()

        if reading_metadata and clean_line == "":
            reading_metadata = False
            continue
        if reading_metadata and ":" in clean_line:
            key, value = clean_line.split(":", 1)

            metadata[key.strip()] = value.strip()
        else:
            content_lines.append(line)

    content = "\n".join(content_lines).strip()

    return {
        "file_name": file_path.name,
        "metadata": metadata,
        "content": content,
    }

def load_all_text_documents(folder_path):

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Document folder does not exist: {folder_path}"
        )
    
    documents = []

    for file_path in folder.glob("*.txt"):
        document = load_text_document(file_path)
        documents.append(document)
    
    return documents
