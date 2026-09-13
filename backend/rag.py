import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

DATA_FILE = os.path.join(
    DATA_DIR,
    "knowledge_base.json"
)


document_chunks = []


def ensure_data_directory():
    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )


def save_knowledge_base():

    ensure_data_directory()

    try:

        with open(
            DATA_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                document_chunks,
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception as error:

        print(
            "Knowledge Base save error:",
            error
        )


def load_knowledge_base():

    ensure_data_directory()

    if not os.path.exists(DATA_FILE):

        return

    try:

        with open(
            DATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):

                document_chunks.extend(
                    data
                )

        print(
            f"Knowledge Base loaded: "
            f"{len(document_chunks)} chunks"
        )

    except Exception as error:

        print(
            "Knowledge Base load error:",
            error
        )


def split_text(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    text = text.strip()

    if not text:

        return chunks

    start = 0

    step = chunk_size - overlap

    while start < len(text):

        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        start += step

    return chunks


def add_document(
    text,
    filename
):

    chunks = split_text(text)

    # Remove previous version of the same file
    remove_document(
        filename,
        save=False
    )

    for chunk in chunks:

        document_chunks.append({

            "filename": filename,

            "text": chunk

        })

    save_knowledge_base()

    return len(chunks)


def remove_document(
    filename,
    save=True
):

    global document_chunks

    original_count = len(
        document_chunks
    )

    document_chunks = [

        item

        for item in document_chunks

        if item.get("filename")
        != filename

    ]

    removed = (
        len(document_chunks)
        < original_count
    )

    if removed and save:

        save_knowledge_base()

    return removed


def clear_documents():

    document_chunks.clear()

    save_knowledge_base()


def get_documents():

    documents = []

    for item in document_chunks:

        filename = item.get(
            "filename",
            "Unknown"
        )

        if filename not in documents:

            documents.append(
                filename
            )

    return documents


def get_knowledge_stats():

    documents = get_documents()

    total_characters = sum(

        len(
            item.get(
                "text",
                ""
            )
        )

        for item in document_chunks
    )

    return {

        "documents": len(documents),

        "chunks": len(
            document_chunks
        ),

        "characters": total_characters

    }


def search_documents(
    query,
    top_k=5
):

    if not document_chunks:

        return []

    query = query.strip()

    if not query:

        return []

    texts = [

        item.get(
            "text",
            ""
        )

        for item in document_chunks

    ]

    if not any(texts):

        return []

    vectorizer = TfidfVectorizer(

        stop_words="english",

        lowercase=True

    )

    try:

        matrix = vectorizer.fit_transform(

            texts + [query]

        )

        document_vectors = matrix[:-1]

        query_vector = matrix[-1]

        scores = cosine_similarity(

            query_vector,

            document_vectors

        )[0]

    except ValueError:

        return []

    ranked_indexes = scores.argsort()[::-1]

    results = []

    for index in ranked_indexes:

        score = float(
            scores[index]
        )

        # Do not return unrelated documents
        if score <= 0:

            continue

        results.append({

            "filename":
                document_chunks[index].get(
                    "filename",
                    "Unknown"
                ),

            "text":
                document_chunks[index].get(
                    "text",
                    ""
                ),

            "score":
                score

        })

        if len(results) >= top_k:

            break

    return results


# Load saved Knowledge Base when AURA starts
load_knowledge_base()
