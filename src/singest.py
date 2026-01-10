import os
import re
import pandas as pd
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def get_vector_db(book_path, embeddings):
    """
    Member 1 Logic: Hybrid approach.
    Uses Semantic splitting first, but forces a Character-wise 
    split if the chunk is too large for the model context.
    """
    with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
        novel_text = f.read()

    # Hard safety limit to prevent the context length error
    character_safety_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )

    # Your original semantic logic
    semantic_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95 
    )
    
    # Split by double newlines first to keep the semantic model stable
    pre_split_paragraphs = [p for p in novel_text.split('\n\n') if len(p.strip()) > 0]
    docs = []
    current_chapter = "Unknown/Intro"
    
    print(f"🔨 Hybrid Processing: {len(pre_split_paragraphs)} paragraphs...")

    for paragraph in pre_split_paragraphs:
        try:
            # 1. Attempt Semantic split
            semantic_chunks = semantic_splitter.split_text(paragraph)
            
            for chunk in semantic_chunks:
                # 2. Character-wise safety check
                # If a semantic chunk is > 1000 chars, it will crash Ollama.
                # In that case, we split it character-wise.
                if len(chunk) > 1000:
                    safe_chunks = character_safety_splitter.split_text(chunk)
                else:
                    safe_chunks = [chunk]

                for final_text in safe_chunks:
                    # Maintain your chapter detection regex logic
                    chapter_match = re.search(r'(Chapter|CHAPTER|CHAP\.)\s+([IVXLCDM\d]+|[A-Za-z ]+)', final_text[:200])
                    if chapter_match:
                        current_chapter = chapter_match.group(0).strip()
                    
                    docs.append(Document(
                        page_content=final_text,
                        metadata={"chapter": current_chapter}
                    ))
        except Exception:
            # Skip corrupted paragraphs and keep moving
            continue

    return Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        collection_name=f"temp_{int(pd.Timestamp.now().timestamp())}"
    )