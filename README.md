# Socratic AI — Basic Socratic Dialogue Simulation

**Live Demo: https://socrates-project-pi.vercel.app//**

---

## Features

- **Gemini 2.0 Flash** integration with model fallback chain (flash → flash-lite → 1.5-flash)
- **Exponential backoff + jitter** for rate limit handling (both server and client side)
- **NLP preprocessing** via spaCy: tokenization, lemmatization, NER, sentiment analysis, key concept extraction
- **Multi-turn conversation** with session history (keeps last 20 turns)
- **Polished dark UI** with ambient animations and responsive design
- **Structured error handling** for rate limits (429), safety filters, API errors, and timeouts

## Local Setup

### Prerequisites

- Python 3.11 or higher
- A Google Gemini API key — get one free at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/socratic-ai.git
cd socratic-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Or create it manually and add your Gemini API key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

### 6. Run the development server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser at **http://localhost:8000**

> The FastAPI app serves the frontend HTML directly, so no separate frontend server is needed.
