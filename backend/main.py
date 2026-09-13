import io
import os
import time

from fastapi import (
    FastAPI,
    UploadFile,
    File
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from pydantic import (
    BaseModel,
    Field
)

from dotenv import load_dotenv

from openai import OpenAI

from pypdf import PdfReader

from rag import (
    add_document,
    search_documents,
    get_documents,
    clear_documents,
    get_knowledge_stats,
    remove_document
)


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()

api_key = os.getenv(
    "OPENAI_API_KEY"
)

if not api_key:

    raise ValueError(
        "OPENAI_API_KEY is missing "
        "from the .env file."
    )


client = OpenAI(
    api_key=api_key
)


# --------------------------------------------------
# APP
# --------------------------------------------------

app = FastAPI(
    title="AURA AI",
    version="10.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MAX_FILE_SIZE = 10 * 1024 * 1024


ALLOWED_TYPES = [

    "application/pdf",

    "text/plain"

]


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ChatRequest(BaseModel):

    message: str

    history: list = Field(
        default_factory=list
    )


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():

    return {

        "status":
            "AURA AI is running",

        "version":
            "10.0"

    }


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
def health():

    return {

        "status":
            "healthy"

    }


# --------------------------------------------------
# DOCUMENT LIST
# --------------------------------------------------

@app.get("/documents")
def documents():

    return {

        "documents":
            get_documents()

    }


# --------------------------------------------------
# KNOWLEDGE BASE STATISTICS
# --------------------------------------------------

@app.get("/documents/stats")
def document_stats():

    return get_knowledge_stats()


# --------------------------------------------------
# IMPORTANT:
# CLEAR ROUTE MUST COME BEFORE
# /documents/{filename}
# --------------------------------------------------

@app.delete("/documents/clear")
def clear_knowledge_base():

    try:

        clear_documents()

        return {

            "success":
                True,

            "message":
                "Knowledge Base cleared successfully."

        }

    except Exception as error:

        return {

            "success":
                False,

            "message":
                f"Could not clear Knowledge Base: {error}"

        }


# --------------------------------------------------
# DELETE ONE DOCUMENT
# --------------------------------------------------

@app.delete("/documents/{filename}")
def delete_document(
    filename: str
):

    try:

        success = remove_document(
            filename
        )

        if success:

            return {

                "success":
                    True,

                "message":
                    f"Document '{filename}' deleted successfully."

            }

        return {

            "success":
                False,

            "message":
                f"Document '{filename}' not found."

        }

    except Exception as error:

        return {

            "success":
                False,

            "message":
                f"Delete error: {error}"

        }


# --------------------------------------------------
# UPLOAD DOCUMENT
# --------------------------------------------------

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...)
):

    if not file.filename:

        return {

            "success":
                False,

            "message":
                "No file selected."

        }


    if file.content_type not in ALLOWED_TYPES:

        return {

            "success":
                False,

            "message":
                "Only PDF and TXT files are supported."

        }


    try:

        file_bytes = await file.read()


        if len(file_bytes) > MAX_FILE_SIZE:

            return {

                "success":
                    False,

                "message":
                    "File is too large. Maximum size is 10 MB."

            }


        extracted_text = ""


        # ------------------------------------------
        # PDF
        # ------------------------------------------

        if file.content_type == "application/pdf":

            pdf_file = io.BytesIO(
                file_bytes
            )

            reader = PdfReader(
                pdf_file
            )


            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:

                    extracted_text += (

                        page_text

                        + "\n"

                    )


        # ------------------------------------------
        # TXT
        # ------------------------------------------

        elif file.content_type == "text/plain":

            extracted_text = file_bytes.decode(

                "utf-8",

                errors="ignore"

            )


        # ------------------------------------------
        # VALIDATE TEXT
        # ------------------------------------------

        if not extracted_text.strip():

            return {

                "success":
                    False,

                "message":
                    "No readable text was found in this document."

            }


        # ------------------------------------------
        # ADD TO RAG
        # ------------------------------------------

        chunk_count = add_document(

            extracted_text,

            file.filename

        )


        return {

            "success":
                True,

            "message":
                f"Document '{file.filename}' uploaded successfully.",

            "filename":
                file.filename,

            "chunks":
                chunk_count

        }


    except Exception as error:

        return {

            "success":
                False,

            "message":
                f"Upload error: {error}"

        }


# --------------------------------------------------
# CHAT
# --------------------------------------------------

@app.post("/chat")
def chat(
    data: ChatRequest
):

    start_time = time.time()


    try:

        message = data.message.strip()


        if not message:

            return {

                "reply":
                    "Please enter a message."

            }


        # ------------------------------------------
        # RAG SEARCH
        # ------------------------------------------

        relevant_chunks = search_documents(

            message,

            top_k=5

        )


        document_context = ""


        if relevant_chunks:

            document_context = (

                "\n\n"
                "RELEVANT INFORMATION "
                "FROM UPLOADED DOCUMENTS:\n"

            )


            for number, chunk in enumerate(

                relevant_chunks,

                start=1

            ):

                document_context += (

                    f"\n"
                    f"[Source {number}: "
                    f"{chunk['filename']}]\n"

                )

                document_context += (

                    chunk["text"]

                    + "\n"

                )


        # ------------------------------------------
        # CONVERSATION HISTORY
        # ------------------------------------------

        conversation = []


        recent_history = (

            data.history[-10:]

        )


        for item in recent_history:

            if (

                isinstance(item, dict)

                and "role" in item

                and "content" in item

            ):

                role = item["role"]

                content = item["content"]


                if role in [

                    "user",

                    "assistant"

                ]:

                    conversation.append({

                        "role":
                            role,

                        "content":
                            content

                    })


        conversation.append({

            "role":
                "user",

            "content":
                message

        })


        # ------------------------------------------
        # AURA INSTRUCTIONS
        # ------------------------------------------

        instructions = (

            "You are AURA, a helpful, "
            "intelligent and professional AI assistant. "

            "Answer clearly and use simple language. "

            "You have access to an uploaded-document "
            "Knowledge Base. "

            "When relevant document information is "
            "provided, use it to answer the user's "
            "question. "

            "Never invent facts from uploaded documents. "

            "If a question specifically asks about "
            "the uploaded documents and the provided "
            "document information does not contain "
            "the answer, clearly say that the information "
            "was not found in the uploaded documents. "

            "For normal general questions that are not "
            "about the uploaded documents, answer "
            "normally using your general knowledge. "

            "Use headings, bullet points and examples "
            "when they improve readability."

        )


        if document_context:

            instructions += document_context


        # ------------------------------------------
        # OPENAI
        # ------------------------------------------

        response = client.responses.create(

            model="gpt-5.6-luna",

            instructions=instructions,

            input=conversation

        )


        reply = response.output_text


        # ------------------------------------------
        # SOURCES
        # ------------------------------------------

        sources = []

        seen_sources = set()


        for chunk in relevant_chunks:

            filename = chunk["filename"]


            if filename not in seen_sources:

                sources.append(
                    filename
                )

                seen_sources.add(
                    filename
                )


        # ------------------------------------------
        # RESPONSE TIME
        # ------------------------------------------

        response_time = round(

            time.time()
            - start_time,

            2

        )


        source_text = ""


        if sources:

            source_text = (

                "\n\n"
                "📚 **Sources used:**\n"

            )


            for filename in sources:

                source_text += (

                    f"📄 {filename}\n"

                )


        return {

            "reply":
                reply
                + source_text,

            "sources":
                sources,

            "response_time":
                response_time

        }


    except Exception as error:

        return {

            "reply":
                "AURA AI Error: "
                + str(error),

            "sources":
                [],

            "response_time":
                round(
                    time.time()
                    - start_time,
                    2
                )

        }
