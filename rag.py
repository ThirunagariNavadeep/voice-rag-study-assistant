from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


NOTES_DIR = Path("data/notes")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def add_uploaded_pdf(uploaded_file):
    NOTES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = NOTES_DIR / uploaded_file.name

    destination.write_bytes(
        uploaded_file.getvalue()
    )

    return destination


def load_documents(pdf_paths=None):
    documents = []

    if pdf_paths is None:
        pdf_paths = list(
            NOTES_DIR.glob("*.pdf")
        )

    for pdf_path in pdf_paths:

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            continue

        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            text = page.extract_text()

            if not text:
                continue

            text = " ".join(
                text.split()
            ).strip()

            if not text:
                continue

            documents.append(
                {
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number,
                }
            )

    return documents


def chunk_text(
    text,
    chunk_size=450,
    overlap=75,
):
    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_index(pdf_paths=None):
    documents = load_documents(
        pdf_paths
    )

    chunks = []

    for document in documents:

        for chunk in chunk_text(
            document["text"]
        ):

            chunks.append(
                {
                    "text": chunk,
                    "source": document["source"],
                    "page": document["page"],
                }
            )

    if not chunks:
        raise ValueError(
            "No readable PDF content found."
        )

    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index, chunks


def search(
    query,
    index,
    chunks,
    top_k=5,
):
    if index is None or not chunks:
        return []

    query = query.strip()

    if not query:
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    top_k = min(
        top_k,
        len(chunks),
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position < 0:
            continue

        result = dict(
            chunks[index_position]
        )

        result["score"] = float(
            score
        )

        results.append(result)

    return results