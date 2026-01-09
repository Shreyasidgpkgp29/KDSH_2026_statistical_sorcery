import os
import pandas as pd
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_text_splitters import RecursiveCharacterTextSplitter 
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
llm = Ollama(model="mistral", temperature=0) 

# 2. OPTIMIZED VERIFICATION PIPELINE
def run_mini_verification(csv_path, books_folder):
    df = pd.read_csv(csv_path)
    # Standardize the book names from CSV to lowercase
    df["book_name"] = df["book_name"].str.lower().str.strip()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    output_data = []

    # Get a list of all actual files in the directory
    actual_files = [f for f in os.listdir(books_folder) if f.endswith(".txt")]

    for book_name in df["book_name"].unique():
        print(f"\n📘 Processing: {book_name}")
        
        # 1. FIND THE ACTUAL FILENAME (Case-Insensitive)
        # This matches "the count..." from CSV to "The Count..." on disk
        target_filename = next((f for f in actual_files if f.lower() == f"{book_name}.txt"), None)

        if target_filename is None:
            print(f"⚠️ Missing file for: {book_name}")
            print(f"   Searching for: {book_name}.txt")
            print(f"   Files found in data/: {actual_files[:5]}")
            continue
            
        # 2. USE THE CORRECT PATH
        book_path = os.path.join(books_folder, target_filename)
        print(f"✅ Found file: {book_path}")

        # 3. OPEN AND PROCESS
        with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
            novel_text = f.read()

        print(f"   🔨 Chunking and Indexing {target_filename}...")
        chunks = text_splitter.create_documents([novel_text])
        vector_db = Chroma.from_documents(chunks, embeddings)
        
        book_claims = df[df["book_name"] == book_name]

        print(f"   🔍 Checking {len(book_claims)} claims...")
        for _, row in book_claims.iterrows():
            claim = row["content"]
            results = vector_db.similarity_search(claim, k=5) 
            context_text = "\n---\n".join([doc.page_content for doc in results])

            prompt = f"""You are a literary fact-checker. 
               BOOK: {book_name}
               CLAIM: {row['content']}
               EVIDENCE: {context_text}

               Compare the CLAIM to the EVIDENCE:
               1. If the EVIDENCE explicitly disproves the CLAIM, label it 'Contradictory'.
               2. If the EVIDENCE supports or does not mention the CLAIM, label it 'Consistent'.

               Format:
               Label: [Consistent/Contradictory]
               Rationale: [One sentence explanation]"""
            response = llm.invoke(prompt)
            output_data.append({
                "id": row["id"],
                "book_name": book_name,
                "claim": claim,
                "model_response": response
            })
        
        vector_db = None 

    return pd.DataFrame(output_data)

# 3. RUN
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: The directory '{DATA_DIR}' does not exist.")
        exit()

    if os.path.exists(TRAIN_CSV):
        print("✅ Found train.csv. Preparing mini_train.csv...")
        full_df = pd.read_csv(TRAIN_CSV)
        
        # --- FIX: Also lowercase here so the logic remains consistent ---
        full_df["book_name"] = full_df["book_name"].str.lower().str.strip()
        
        mini_df = full_df.head(10)
        mini_df.to_csv(MINI_CSV, index=False)
    else:
        print(f"❌ Error: {TRAIN_CSV} not found!")
        exit()

    print("🚀 Starting verification pipeline...")
    results = run_mini_verification(MINI_CSV, DATA_DIR)
    
    if not results.empty:
        results.to_csv(RESULTS_CSV, index=False)
        print(f"\n🎉 Completed! Results saved to {RESULTS_CSV}")
    else:
        print("\n❌ Pipeline finished but no results were generated. Check your filenames!")