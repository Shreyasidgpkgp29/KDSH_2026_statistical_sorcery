import os
import pandas as pd
# Use standard sqlite3 if pysqlite3-binary isn't installed yet
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# --- CONFIG ---
DATA_DIR = "./data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
MINI_CSV = os.path.join(DATA_DIR, "mini_train.csv")
RESULTS_CSV = "verification_results.csv"

# 1. SPEED FIX: Use 'all-minilm' for 5x faster testing on Day 1
# You can switch back to 'mxbai-embed-large' for the final submission
embeddings = OllamaEmbeddings(model="all-minilm", base_url="http://localhost:11434")
llm = Ollama(model="mistral", temperature=0) 

def run_mini_verification(csv_path, books_folder):
    df = pd.read_csv(csv_path)
    df["book_name"] = df["book_name"].str.lower().str.strip()
    
    # SPEED FIX: Increased chunk size to 1500 to reduce the total number of vectors
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    output_data = []

    actual_files = [f for f in os.listdir(books_folder) if f.endswith(".txt")]

    for book_name in df["book_name"].unique():
        target_filename = next((f for f in actual_files if f.lower() == f"{book_name}.txt"), None)

        if target_filename is None:
            continue
            
        book_path = os.path.join(books_folder, target_filename)
        print(f"✅ Found file: {book_path}")

        with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
            # SPEED FIX: Only read the first 20% of the book for Day 1 testing
            # This ensures your pipeline works without waiting 20 minutes
            novel_text = f.read(500000) 

        print(f"   🔨 Indexing {target_filename} (Partial Load for Speed)...")
        chunks = text_splitter.create_documents([novel_text])
        vector_db = Chroma.from_documents(chunks, embeddings)
        
        book_claims = df[df["book_name"] == book_name]

        print(f"   🔍 Checking {len(book_claims)} claims...")
        for _, row in book_claims.iterrows():
            results = vector_db.similarity_search(row["content"], k=2) 
            context_text = "\n---\n".join([doc.page_content for doc in results])

            prompt = f"""BOOK: {book_name}\nCLAIM: {row['content']}\nEVIDENCE: {context_text}\n
            Label: [Consistent/Contradictory]\nRationale: [One sentence]"""

            response = llm.invoke(prompt)
            output_data.append({"id": row["id"], "model_response": response})
        
        vector_db = None 

    return pd.DataFrame(output_data)

if __name__ == "__main__":
    # Ensure you only test on books you actually have in the folder
    full_df = pd.read_csv(TRAIN_CSV)
    full_df["book_name"] = full_df["book_name"].str.lower().str.strip()
    available_books = [f.lower().replace(".txt", "") for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    
    mini_df = full_df[full_df["book_name"].isin(available_books)].head(5)
    mini_df.to_csv(MINI_CSV, index=False)

    print("🚀 Starting High-Speed Verification...")
    results = run_mini_verification(MINI_CSV, DATA_DIR)
    results.to_csv(RESULTS_CSV, index=False)
    print(f"🎉 Done! Results saved to {RESULTS_CSV}")