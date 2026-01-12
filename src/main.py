import os
import pandas as pd
import sys
import re
import numpy as np
import pathway as pw
import pathway.stdlib.ml.index as pw_index

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaEmbeddings
import singest
from prompts import verify_claim

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

INPUT_FILE = "Dataset/test.csv"
BOOKS_DIR = "Dataset/Books"
OUTPUT_FILE = "results.csv"

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

def main():
    print("Starting KDSH Handshake Pipeline...")

    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    df["book_name_std"] = df["book_name"].str.lower().str.strip()

    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        processed_ids = set(existing_df["Story ID"].unique())
        print(f"Resuming: {len(processed_ids)} claims already processed.")

    actual_files = [f for f in os.listdir(BOOKS_DIR) if f.endswith(".txt")]
    unique_books = df["book_name_std"].unique()

    for book_name in unique_books:
        book_claims = df[(df["book_name_std"] == book_name) & (~df["id"].isin(processed_ids))]
        
        if book_claims.empty:
            continue

        target_filename = next((f for f in actual_files if f.lower() == f"{book_name}.txt"), None)
        if not target_filename:
            print(f"Skipping: {book_name} (No .txt file)")
            continue       

        book_path = os.path.join(BOOKS_DIR, target_filename)
        print(f"Processing: {target_filename} ({len(book_claims)} new claims)")

        vector_db = singest.get_vector_db(book_path, embeddings)

        current_book_results = []

        for _, row in book_claims.iterrows():

            query_table = pw.debug.table_from_rows(
                rows=[(row["content"],)],
                schema=pw.schema_from_dict({"text": str})
            )

            query_embedded = query_table.select(
                vector=pw.apply(
                    lambda x: normalize(embeddings.embed_query(x)),
                    pw.this.text
                )
            )

            search_results = vector_db.get_nearest_items(
                query_embedded.vector, 
                k=5
            )
            results_df = pw.debug.table_to_pandas(search_results)

            context_tuple = results_df["text"].iloc[0]
            context_text = "\n---\n".join(context_tuple)

            label, rationale = verify_claim(context_text, row["content"])

            current_book_results.append({
                "Story ID": row["id"],
                "Prediction": label,
                "Rationale": rationale
            })   

        if current_book_results:
            checkpoint_df = pd.DataFrame(current_book_results)
            checkpoint_df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not os.path.exists(OUTPUT_FILE))
            print(f"Saved results for {target_filename} to {OUTPUT_FILE}")

        del vector_db 

    print(f"\nPipeline Complete! Final results in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()