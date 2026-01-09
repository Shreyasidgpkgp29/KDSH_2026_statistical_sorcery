import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter # ⚡ 100x Faster
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# --- CONFIG ---
DATA_DIR = "./data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
MINI_CSV = os.path.join(DATA_DIR, "mini_train.csv")
RESULTS_CSV = "verification_results.csv"

# 1. EMBEDDINGS + LLM
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
llm = Ollama(model="mistral", temperature=0) # temperature=0 for fact-checking!

# 2. OPTIMIZED VERIFICATION PIPELINE
def run_mini_verification(csv_path, books_folder):
    df = pd.read_csv(csv_path)
    # Use Recursive splitter for speed in a hackathon setting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    output_data = []

    for book_name in df["book_name"].unique():
        print(f"\n📘 Processing: {book_name}")
        book_path = os.path.join(books_folder, f"{book_name}.lower().txt")

        if not os.path.exists(book_path):
            print(f"⚠️ Missing file: {book_path}")
            continue

        with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
            novel_text = f.read()
        novel_text = novel_text[:15000]  # limit to first 15k characters

        # Fast Chunking
        chunks = text_splitter.create_documents([novel_text])

        # Create Vector DB (In-memory for speed)
        vector_db = Chroma.from_documents(chunks, embeddings)
        book_claims = df[df["book_name"] == book_name]

        for _, row in book_claims.iterrows():
            claim = row["content"]
            # Increased k to 3 to catch more context
            results = vector_db.similarity_search(claim, k=3) 
            
            context_text = "\n---\n".join([doc.page_content for doc in results])

            prompt = f"""You are a literary fact-checker. 
            BOOK: {book_name}
            CLAIM: {row['content']}
            EVIDENCE: {context_text}

            Does the EVIDENCE support or contradict the CLAIM?
            If the evidence doesn't mention the claim, assume Consistent unless it explicitly conflicts.

            Format:
            Label: [Consistent/Contradictory]
            Rationale: [One sentence explanation citing the evidence]"""

            response = llm.invoke(prompt)
            output_data.append({
                "id": row["id"],
                "book_name": book_name,
                "claim": claim,
                "model_response": response
            })
        
        # Explicitly clear memory
        vector_db = None 

    return pd.DataFrame(output_data)

# 3. RUN
if __name__ == "__main__":
    results = run_mini_verification(MINI_CSV, DATA_DIR)
    results.to_csv(RESULTS_CSV, index=False)
    print("\n🎉 Hackathon pipeline completed successfully!")