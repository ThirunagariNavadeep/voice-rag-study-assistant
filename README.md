# 🎙️ Voice RAG Study Assistant

An English voice-enabled Retrieval-Augmented Generation (RAG) application that allows users to upload PDF study material, ask questions using voice or text, retrieve relevant information from the uploaded documents, and receive grounded answers both on screen and through voice.

## 🚀 Overview

The Voice RAG Study Assistant combines document retrieval, local LLM inference, speech recognition, and text-to-speech into a single study assistant.

Instead of answering questions purely from the language model's general knowledge, the system first searches the uploaded PDF documents and provides the retrieved context to the LLM.

This helps keep answers grounded in the user's study material.

### Core workflow

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Text Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Index
    ↓
User Question
    ↓
Semantic Retrieval
    ↓
Qwen3
    ↓
Grounded Answer
    ├──→ Screen
    └──→ Kokoro TTS → Voice

✨ Features
📄 PDF-based Knowledge Base
Upload one or multiple PDF documents.
Extract text page-by-page.
Split documents into overlapping chunks.
Generate semantic embeddings.
Store embeddings in FAISS.
Display the documents currently included in the knowledge base.
🔎 Retrieval-Augmented Generation

The application retrieves relevant sections of the uploaded documents before generating an answer.

The LLM is instructed to:

Use retrieved document content as the primary source.
Avoid unsupported claims.
Clearly indicate when information cannot be found in the uploaded material.
Provide document and page references.
🎤 Voice Questions

Users can ask questions using their microphone.

Voice
 ↓
Whisper
 ↓
Text
 ↓
RAG
 ↓
Answer

The transcription is displayed before the question is submitted.

Users can:

✅ Answer
🗑️ Discard
🎤 Re-record
🤖 Local LLM

The application uses Qwen3 through Ollama for answer generation.

The model runs locally, allowing the core question-answering workflow to operate without sending study documents to a hosted LLM API.

🔊 Text-to-Speech

Generated answers are converted into spoken responses using Kokoro TTS.

Markdown formatting such as:

**
##
-

is removed before speech generation so formatting symbols are not spoken aloud.

📝 Exam-oriented Answers

The assistant understands instructions such as:

Explain backpropagation for 2 marks.
Explain backpropagation for 5 marks.
Explain backpropagation for 10 marks.

It also handles word-count requirements:

Explain backpropagation in 100 words.

or:

Explain backpropagation for 5 marks in 150 words.
💬 Conversation Context

Recent questions and answers are retained so follow-up questions can use conversational context.

Example:

User:
What is backpropagation?

Assistant:
[Answer]

User:
Why is it required?

Assistant:
[Uses the previous conversation context]
📚 Source Attribution

Answers include the PDF and page numbers used during retrieval.

Example:

Sources:
- machine_learning.pdf (Page 113)
- machine_learning.pdf (Page 116)
🛠️ Tech Stack
Component	Technology
Language	Python
UI	Streamlit
LLM	Qwen3
Local inference	Ollama
RAG	Retrieval-Augmented Generation
Embeddings	Sentence Transformers
Vector search	FAISS
PDF processing	pypdf
Speech-to-text	Whisper
Text-to-speech	Kokoro
Audio	SoundFile
Numerical processing	NumPy
📁 Project Structure
voice-rag-study-assistant/
│
├── app/
│   ├── agent.py
│   ├── rag.py
│   ├── speech.py
│   ├── tools.py
│   ├── tts.py
│   └── ui.py
│
├── data/
│   └── notes/
│       └── .gitkeep
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── run.bat
⚙️ Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/voice-rag-study-assistant.git
cd voice-rag-study-assistant
2. Create a virtual environment
Windows
python -m venv .venv

Activate it:

.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
🤖 Install Ollama

Install Ollama from its official website.

After installation, verify:

ollama --version

Pull the Qwen3 model:

ollama pull qwen3:8b

Verify that the model is available:

ollama list

You should see:

qwen3:8b
🎙️ Whisper Setup

The application uses Whisper for English speech recognition.

The first execution may download the selected Whisper model.

The current implementation uses:

whisper.load_model("base")

For a faster but potentially less accurate model, the configuration can be changed to:

whisper.load_model("tiny")
▶️ Run the Application

From the project root:

python -m streamlit run app/ui.py

The application will open in your browser.

Typical local address:

http://localhost:8501

Alternatively, on Windows:

run.bat
📖 How to Use
Step 1 — Upload study material

Upload one or more PDF files.

Example:

machine_learning.pdf
deep_learning.pdf
neural_networks.pdf
Step 2 — Build the knowledge base

Click:

Build Knowledge Base

The application will:

Read the PDFs.
Extract page-level text.
Split the text into chunks.
Generate embeddings.
Build the FAISS index.
Step 3 — Ask a question

You can either:

Voice

Click the microphone and ask:

What is backpropagation?

Review the transcription and click:

Answer
Text

Type the question in the text input.

🎓 Exam Question Examples
Normal question
What is backpropagation?
2-mark question
What is backpropagation for 2 marks?
5-mark question
Explain backpropagation for 5 marks.
10-mark question
Explain backpropagation for 10 marks.
Word-count requirement
Explain backpropagation in 100 words.
Combined requirement
Explain backpropagation for 5 marks in 150 words.
🧠 RAG Pipeline

The retrieval pipeline works as follows:

Uploaded PDF
     ↓
pypdf
     ↓
Page-level text
     ↓
Overlapping chunks
     ↓
all-MiniLM-L6-v2
     ↓
Embeddings
     ↓
FAISS

When the user asks a question:

Question
   ↓
Question embedding
   ↓
FAISS similarity search
   ↓
Top relevant chunks
   ↓
Qwen3
   ↓
Grounded answer
🔐 Grounding Strategy

The assistant is explicitly instructed not to fabricate information when the uploaded documents do not contain enough information.

For unsupported questions, the expected behavior is:

I couldn't find enough information about
this in the uploaded study material.

This is important because the application is designed as a document-grounded study assistant, rather than a general-purpose chatbot.

🎤 Voice Pipeline
Microphone
    ↓
Audio
    ↓
Whisper
    ↓
Transcription
    ↓
User Review
    ↓
RAG Question Answering
    ↓
Kokoro TTS
    ↓
response.wav
    ↓
Browser Audio Player

The transcription can be discarded or re-recorded before sending it to the RAG pipeline.

⚡ Performance

The application includes optional performance measurements for the major AI stages.

Example:

[PERF] RAG retrieval: 0.03s
[PERF] Qwen3 generation: 4.82s
[PERF] Kokoro TTS: 2.41s

This makes it easier to identify performance bottlenecks during development.

🧪 Testing Checklist

Before deployment, verify:

Knowledge Base
 Upload PDF
 Build knowledge base
 Multiple PDFs
 Display uploaded documents
RAG
 Relevant question
 Correct retrieval
 Source/page references
 Unsupported question handling
Voice
 Record question
 Transcribe question
 Answer transcription
 Discard
 Re-record
Answer
 Screen answer
 Voice answer
 Markdown cleaned for TTS
 Audio persistence
 Autoplay attempt
Exam requirements
 Marks-based answers
 Word-count requirements
 Combined marks + word count
🚧 Known Limitations
Browser autoplay policies can prevent automatic audible playback.
Whisper transcription speed depends on the selected model and hardware.
Qwen3 inference speed depends on available CPU/GPU resources.
Very large PDF collections may require more advanced indexing and retrieval strategies.
Scanned/image-only PDFs require OCR support because standard PDF text extraction may not extract their content.
🔮 Future Improvements

Potential future improvements include:

Hybrid keyword + semantic retrieval
Reranking retrieved chunks
Streaming LLM responses
Streaming TTS
OCR for scanned PDFs
Table-aware PDF extraction
Conversation memory improvements
GPU-optimized inference
Cloud deployment
Authentication and multi-user document isolation
Evaluation benchmarks for retrieval and answer quality
💡 Why This Project?

Traditional PDF study workflows require users to manually search through long documents to find answers.

This project combines:

Document Retrieval
+
LLM Reasoning
+
Voice Interaction
+
Text-to-Speech

to create a more natural way to interact with study material.

The key design principle is:

Retrieve first, generate second.

The LLM is given relevant document context before generating the answer, reducing dependence on unsupported model knowledge.

🧩 Technical Challenges
1. Grounded answers

A general LLM can answer using knowledge outside the uploaded documents.

Solution:

Question
 ↓
FAISS retrieval
 ↓
Relevant document chunks
 ↓
Qwen3

The model is instructed to stay within the retrieved context.

2. Markdown in TTS

LLMs naturally generate Markdown formatting.

Sending that directly to TTS can result in symbols being spoken.

Solution:

LLM Markdown
 ↓
TTS text cleaning
 ↓
Kokoro
3. Voice transcription review

Speech recognition can produce incorrect words.

Solution:

Voice
 ↓
Whisper
 ↓
Display transcription
 ↓
Answer / Discard / Re-record

This gives the user control before the question reaches the RAG pipeline.

4. Per-document knowledge base

The application rebuilds the FAISS index from the PDFs selected for the current knowledge-base build instead of blindly searching every PDF stored in the project directory.

This keeps retrieval scoped to the user's selected study material.

📌 Resume Description
Voice RAG Study Assistant

Python, Qwen3, FAISS, Sentence Transformers, Whisper, Kokoro, Streamlit

Built an English voice-enabled RAG application that processes uploaded PDF study material and retrieves semantically relevant content using Sentence Transformer embeddings and FAISS.
Integrated Qwen3 for grounded document-based question answering with support for marks-based and word-count-constrained responses.
Developed an end-to-end speech pipeline using Whisper for speech recognition and Kokoro TTS for spoken responses, with transcription review, discard/re-record controls, and persistent audio playback.
Implemented page-level source attribution and conversational context to improve answer traceability and follow-up question handling.
👨‍💻 Author

Navadeep Thirunagari

GitHub:

https://github.com/ThirunagariNavadeep