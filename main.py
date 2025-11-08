import os
import re
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", 300))
TOP_K = int(os.getenv("TOP_K", 5))

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load movie dataset
def load_dataset(path="data/wiki_movie_plots.csv", limit=300):
    df = pd.read_csv(path)
    df = df[["Title", "Plot"]].dropna().head(limit)
    return df

# Chunk long plots
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Setup Chroma vector store
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=OPENAI_EMBED_MODEL
)
collection = chroma_client.get_or_create_collection(name="movie_plots", embedding_function=embedding_fn)


# Ingest data into chroma db
def build_vector_store():
    if collection.count() > 0:
        print(f" Found {collection.count()} existing documents in Chroma. Skipping rebuild.")
        return

    df = load_dataset()
    ids, texts, metas = [], [], []

    print("...Building vector store...")
    seen_ids = set()

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        title, plot = row["Title"].strip(), row["Plot"]
        chunks = chunk_text(plot, CHUNK_WORDS)

        # Add a global row index to ensure uniqueness
        for i, chunk in enumerate(chunks):
            
            base_id = re.sub(r"[^a-zA-Z0-9]+", "_", title)
            unique_id = f"{base_id}_{idx}_{i}"
            
            if unique_id in seen_ids:
                unique_id = f"{unique_id}_{len(seen_ids)}"
            seen_ids.add(unique_id)

            ids.append(unique_id)
            texts.append(chunk)
            metas.append({"title": title})

    collection.add(documents=texts, metadatas=metas, ids=ids)
    print(f"Stored {len(texts)} chunks in ChromaDB.")


# Retrieve relevant chunks
def retrieve_context(query, k=TOP_K):
    results = collection.query(query_texts=[query], n_results=k)
    contexts = results["documents"][0]
    return contexts


# Generate answer using OpenAI
def generate_answer(movie_name, contexts):
    """
    Generate a detailed answer to the question using retrieved contexts.
    """
    context_str = "\n---\n".join(contexts)
    prompt = f"""
You are a movie expert. Using the following movie plot excerpts, answer the user's question. Your response should:

- Base your answer only on the provided excerpts; do not invent any details.
- Be concise and clear.
- Directly address the question asked.
- Include any relevant characters, events, or themes from the excerpts that help support your answer.
- If the answer is not explicitly in the excerpts, indicate that the information is not available.
- Provide your reasoning in a short explanation of how you formed the answer from the excerpts.

Context:
{context_str}

Movie name: {movie_name}

Provide your final answer (do not include reasoning here).
    """
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


# Generate reasoning
def generate_reasoning(answer, contexts):
    """
    Ask the LLM to generate a concise reasoning explaining how it formed the answer.
    """
    context_str = "\n---\n".join(contexts)
    prompt = f"""
You are a movie expert. You have already generated the following answer based on the retrieved movie plot excerpts:

Answer:
{answer}

Excerpts:
{context_str}

Provide a concise reasoning in 1-2 sentences explaining how you derived this answer from the excerpts. 
Respond only with the reasoning, do not repeat the answer.
    """
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


# Combine answer + reasoning
def generate_answer_with_reasoning(movie_name, contexts):
    answer = generate_answer(movie_name, contexts)
    reasoning = generate_reasoning(answer, contexts)
    output = {
        "answer": answer,
        "contexts": contexts,
        "reasoning": reasoning
    }
    return json.dumps(output, indent=2)


# Run interactive QA loop
if __name__ == "__main__":
    build_vector_store()
    print("\n Movie RAG System ready! Ask questions (type 'exit' to quit)\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        # Retrieve relevant plot excerpts
        contexts = retrieve_context(query, TOP_K)

        # Generate structured answer with reasoning
        answer_json = generate_answer_with_reasoning(query, contexts)

        print("\n Response:\n", answer_json, "\n")
