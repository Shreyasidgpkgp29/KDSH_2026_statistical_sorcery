import os
import pandas as pd
import sys

# SAFETY: Ensure Python looks in the current folder for singest and prompts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use the community version you have installed
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import singest
from prompts import verify_claim

# Path setup (using ../ because you are running from inside src/)
# These point to the data folder in the root directory
INPUT_FILE = "../data/train.csv"
BOOKS_DIR = "../data"
OUTPUT_FILE = "submission.csv"

# SQLite fix for environments where the default sqlite3 is outdated
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

def main():
    print("🚀 Starting KDSH Handshake Pipeline...")
    
    # Initialize using the older class you prefer
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
    
    # Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Check your directory structure.")
        return

    df = pd.read_csv(INPUT_FILE).head(10) # Testing with first 10
    df["book_name_std"] = df["book_name"].str.lower().str.strip()
    
    actual_files = [f for f in os.listdir(BOOKS_DIR) if f.endswith(".txt")]
    output_data = []

    # Process by Unique Book to save time/memory
    for book_name in df["book_name_std"].unique():
        target_filename = next((f for f in actual_files if f.lower() == f"{book_name}.txt"), None)
        
        if not target_filename:
            print(f"⚠️ Skipping: {book_name} (No matching .txt file)")
            continue
            
        book_path = os.path.join(BOOKS_DIR, target_filename)
        print(f"📘 Processing: {target_filename}")
        
        # HANDSHAKE 1: Build Hybrid Vector DB from singest.py
        # This will now process the full book safely using character-wise safety checks
        vector_db = singest.get_vector_db(book_path, embeddings)
        
        book_claims = df[df["book_name_std"] == book_name]
        print(f"  🔍 Checking {len(book_claims)} claims for this book...")

        for _, row in book_claims.iterrows():
            # HANDSHAKE 2: Retrieve the 5 most related chunks
            results = vector_db.similarity_search(row["content"], k=5)
            context_text = "\n---\n".join([doc.page_content for doc in results])
            
            # HANDSHAKE 3: Verify using logic in prompts.py
            # verify_claim returns (label, rationale)
            label, rationale = verify_claim(context_text, row["content"])
            
            output_data.append({
                "id": row["id"],
                "label": label,
                "rationale": rationale
            })
        
        # Cleanup: Crucial for 16GB RAM to prevent "Monte Cristo" from crashing the system
        vector_db.delete_collection()
        print(f"✅ Finished {target_filename}")

    # Save Final Results
    pd.DataFrame(output_data).to_csv(OUTPUT_FILE, index=False)
    print(f"\n🎉 Success! Results saved to {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()