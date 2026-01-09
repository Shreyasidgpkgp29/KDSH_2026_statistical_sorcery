import os
import pandas as pd
import re

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document

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
    df["book_name_std"] = df["book_name"].str.lower().str.strip()
    
    # Initialize Semantic Chunker
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95 
    )
    
    output_data = []
    actual_files = [f for f in os.listdir(books_folder) if f.endswith(".txt")]

    for book_name in df["book_name_std"].unique():
        target_filename = next((f for f in actual_files if f.lower() == f"{book_name}.txt"), None)
        if target_filename is None: continue
            
        book_path = os.path.join(books_folder, target_filename)
        print(f"✅ Processing: {target_filename}")

        with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
            # REDUCED SIZE: Lowering to 200k to avoid context overflow
            novel_text = f.read(200000) 

        print(f"   🔨 Pre-processing text to avoid context length error...")
        # FIX: Split by double newlines first so the SemanticChunker doesn't overload Ollama
        pre_split_texts = [p for p in novel_text.split('\n\n') if len(p.strip()) > 0]
        
        docs = []
        current_chapter = "Unknown/Intro"
        
        print(f"   🔨 Generating Semantic Chunks for {len(pre_split_texts)} sections...")
        for section in pre_split_texts:
            try:
                # Process each section semantically
                section_chunks = text_splitter.split_text(section)
                for chunk_text in section_chunks:
                    # Detect chapter headings
                    chapter_match = re.search(r'(Chapter|CHAPTER|CHAP\.)\s+([IVXLCDM\d]+|[A-Za-z ]+)', chunk_text[:200])
                    if chapter_match:
                        current_chapter = chapter_match.group(0).strip()
                    
                    docs.append(Document(
                        page_content=chunk_text,
                        metadata={"chapter": current_chapter}
                    ))
            except Exception as e:
                # If a section is still too big, skip it or log it
                continue

        # Create temporary Vector Store
        vector_db = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings,
            collection_name=f"temp_{int(pd.Timestamp.now().timestamp())}"
        )
        
        book_claims = df[df["book_name_std"] == book_name]

        print(f"   🔍 Checking {len(book_claims)} claims...")
        for _, row in book_claims.iterrows():
            results = vector_db.similarity_search(row["content"], k=5) 
            
            top_chapter = results[0].metadata['chapter'] if results else "Not Found"
            context_text = "\n---\n".join([doc.page_content for doc in results])

            prompt = f"""BOOK: {row['book_name']}\nCLAIM: {row['content']}\nEVIDENCE: {context_text}\n
            Label: [Consistent/Contradictory]\nRationale: [One sentence]"""
            
            response = llm.invoke(prompt)
            
            output_data.append({
                "id": row.get("id"),
                "novel_name": row.get("book_name"),
                "char": row.get("char"), 
                "claim": row["content"],
                "chapter_identified": top_chapter,
                "model_response": response
            })
        
        vector_db.delete_collection()
        vector_db = None 

    return pd.DataFrame(output_data)

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: {DATA_DIR} directory missing.")
        exit()

    if os.path.exists(TRAIN_CSV):
        full_df = pd.read_csv(TRAIN_CSV)
        mini_df = full_df.head(10)
        mini_df.to_csv(MINI_CSV, index=False)
    else:
        print(f"❌ Error: {TRAIN_CSV} not found!")
        exit()

    print("🚀 Starting fixed pipeline...")
    results = run_mini_verification(MINI_CSV, DATA_DIR)
    
    if not results.empty:
        results.to_csv(RESULTS_CSV, index=False)
        print(f"\n🎉 Success! Results saved to {RESULTS_CSV}")