import pathway as pw
import re
import pandas as pd
import numpy as np 
import pathway.stdlib.ml.index as pw_index

def character_safety_split(text, chunk_size=1000, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def semantic_split(paragraph, embeddings, threshold=0.8):
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    if len(sentences) <= 1:
        return [paragraph]

    vecs = embeddings.embed_documents(sentences)
    chunks, current = [], [sentences[0]]

    for i in range(1, len(sentences)):
        v1, v2 = vecs[i - 1], vecs[i]
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        sim = np.dot(v1, v2) / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0

        if sim < threshold:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    chunks.append(" ".join(current))
    return chunks

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def get_vector_db(book_path, embeddings):
    with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
        novel_text = f.read()

    pre_split_paragraphs = [p for p in novel_text.split("\n\n") if len(p.strip()) > 0]
    raw_data = []
    current_chapter = "Prologue/Introduction" 

    print(f"🔨 Hybrid Processing: {len(pre_split_paragraphs)} paragraphs...")

    for paragraph in pre_split_paragraphs:
        try:
            semantic_chunks = semantic_split(paragraph, embeddings)
            for chunk in semantic_chunks:
                if len(chunk.strip()) < 200: continue
                safe_chunks = character_safety_split(chunk) if len(chunk) > 1000 else [chunk]

                for final_text in safe_chunks:
                    chapter_match = re.search(r'(Chapter|CHAPTER|CHAP\.)\s+([IVXLCDM\d]+|[A-Za-z ]+)', final_text[:300])
                    if chapter_match:
                        current_chapter = chapter_match.group(0).strip()
                    
                    raw_data.append({"text": final_text, "chapter": current_chapter})
        except Exception as e:
            continue

    texts = [item["text"] for item in raw_data]
    print(f"Sending {len(texts)} chunks to Ollama in one massive batch...")
    
    all_vectors = embeddings.embed_documents(texts)

    final_rows = []
    for i, (item, vec) in enumerate(zip(raw_data, all_vectors)):
        final_rows.append((
            f"row_{i}", 
            i, 
            item["text"], 
            item["chapter"], 
            normalize(np.array(vec)).tolist()
        ))

    class FinalSchema(pw.Schema):
        chunk_id: str
        seq_num: int
        text: str
        chapter: str
        vector: list

    docs_table = pw.debug.table_from_rows(rows=final_rows, schema=FinalSchema)

    vector_index = pw_index.KNNIndex(
        docs_table.vector,
        docs_table,
        n_dimensions=len(all_vectors[0])
    )

    print(f"Indexing complete for {len(texts)} chunks.")
    return vector_index