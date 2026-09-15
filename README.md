# AURA AI

AURA AI is an intelligent AI assistant built with Python, FastAPI, JavaScript, OpenAI API, and Retrieval-Augmented Generation (RAG).

## Features

- AI-powered chat
- Conversation memory
- PDF and TXT document upload
- Multi-document knowledge base
- Retrieval-Augmented Generation (RAG)
- TF-IDF document retrieval
- Cosine similarity search
- Knowledge Base statistics
- Document deletion
- Clear Knowledge Base
- Voice input
- Text-to-speech
- Stop generation
- New Chat
- Clear Chat
- Modern responsive UI
- Persistent knowledge base
- REST API backend
- File validation
- Error handling

## Technology Stack

### Frontend

- HTML5
- CSS3
- JavaScript
- Web Speech API
- Local Storage

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic

### AI

- OpenAI API
- GPT-based conversational intelligence

### RAG

- TF-IDF Vectorization
- Cosine Similarity
- Text Chunking
- PDF Text Extraction
- Persistent JSON Knowledge Base

## How AURA AI Works

1. User enters a question.
2. AURA receives the message through the web interface.
3. Recent conversation history is sent to the backend.
4. The backend searches the knowledge base when relevant.
5. TF-IDF and cosine similarity find relevant document content.
6. Relevant information is provided to the AI model.
7. The AI generates the response.
8. AURA displays the answer.

## Project Structure

AURA/
├── backend/
│   ├── main.py
│   ├── rag.py
│   ├── .env
│   └── data/
├── frontend/
│   └── index.html
├── .gitignore
├── requirements.txt
└── README.md

## Installation

Clone the repository:

git clone https://github.com/santoshsiddharaddi/AURA-AI.git

Enter the project:

cd AURA-AI

Install dependencies:

pip3 install -r requirements.txt

## Environment Setup

Create:

backend/.env

Add:

OPENAI_API_KEY=your_api_key_here

Never upload your API key to GitHub.

## Run Backend

cd backend
python3 -m uvicorn main:app --reload

Backend:

http://127.0.0.1:8000

Health check:

http://127.0.0.1:8000/health

## Run Frontend

Open another Terminal:

cd ~/Desktop/AURA/frontend
python3 -m http.server 5500

Then open:

http://127.0.0.1:5500/index.html

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | API information |
| GET | /health | Health check |
| GET | /documents | List documents |
| GET | /documents/stats | Knowledge Base statistics |
| DELETE | /documents/clear | Clear Knowledge Base |
| DELETE | /documents/{filename} | Delete document |
| POST | /upload | Upload PDF/TXT |
| POST | /chat | Chat with AURA |

## Security

- API key stored in environment variables
- .env excluded from Git
- File type validation
- Maximum upload size
- Empty message validation
- Error handling

## Learning Outcomes

- Python
- FastAPI
- REST APIs
- JavaScript
- HTML
- CSS
- OpenAI API
- Retrieval-Augmented Generation
- TF-IDF
- Cosine similarity
- NLP concepts
- PDF processing
- Git
- GitHub
- Software architecture
- API communication

## Future Improvements

- User authentication
- Database integration
- Vector database
- Semantic embeddings
- Streaming responses
- Cloud deployment
- Web search
- Multi-user support
- Advanced AI agents
- Mobile application

## Author

**Santosh Siddharaddi**

AURA AI is an academic and portfolio project demonstrating practical skills in Artificial Intelligence, Python, Web Development, APIs, and Retrieval-Augmented Generation.

## License

This project is intended for educational and portfolio purposes.
