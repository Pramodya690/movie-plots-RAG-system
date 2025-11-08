
# Movie RAG System

A lightweight **RAG system** for answering questions about movie plots. This system retrieves relevant movie plot excerpts and generates structured answers with reasoning.

---

## Features

- Load and preprocess movie plot data from CSV.
- Chunk long plot texts for better retrieval.
- Store embeddings in **ChromaDB** for efficient retrieval.
- Retrieve the most relevant chunks based on user queries.
- Generate answers and reasoning using **OpenAI GPT models**.
- Interactive CLI-based question-answering loop.

---

## Tech Stack & Libraries

- **Python 3.10+**
- **Pandas** – Data processing
- **tqdm** – Progress bars for ingestion
- **python-dotenv** – Load environment variables from `.env`
- **ChromaDB** – Vector store for embeddings
- **OpenAI API** – LLM and embeddings
- **Regex (re)** – Clean and create unique IDs for chunks
- **JSON** – Structured output for answers with reasoning

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/movie-rag-system.git
cd movie-rag-system
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a `.env` file

Create a `.env` file in the project root with your API keys and settings:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
CHROMA_PERSIST_DIR=chroma_db
CHUNK_WORDS=300
TOP_K=5
```

### 4. Add dataset

Place your movie dataset CSV (`wiki_movie_plots.csv`) in the `data/` folder.
The system expects columns: `Title` and `Plot`

---

## Running the System

```bash
python main.py
```

You can then interactively ask questions about movies:

```
Your question: Who attacked Boone's daughter?
```

The system will return a structured JSON response containing:

* `answer` – the generated answer
* `contexts` – the retrieved plot excerpts
* `reasoning` – short reasoning explaining how the answer was formed

Type `exit` or `quit` to leave the loop.

---

## Notes

* The system uses the first 300 rows of the dataset by default.
* ChromaDB stores embeddings locally in `CHROMA_PERSIST_DIR`.

---

