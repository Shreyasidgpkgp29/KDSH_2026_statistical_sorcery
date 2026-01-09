import pandas as pd
import os

# ==========================================
# PART 1: THE "MOCK" MODULES (Temporary)
# ==========================================
# Right now, these functions live here. 
# Later, you will DELETE this section and import them from your friends' files.

def get_context_from_ingest(book_name, claim):
    """
    TEMPORARY PLACEHOLDER for Member 1 (Ingest).
    Acts as if it searched the book and found text.
    """
    # M1's logic will go here later. For now, just return a dummy string.
    return f"[MOCK CONTEXT]: Found info in {book_name} related to: {claim[:30]}..."

def verify_with_logic(context, claim):
    """
    TEMPORARY PLACEHOLDER for Member 3 (Logic).
    Acts as if it checked the claim against the context.
    """
    # M3's logic will go here later. For now, return a random guess.
    return "Consistent", "The evidence in the text supports the claim perfectly."

# ==========================================
# PART 2: THE PIPELINE (Your Job)
# ==========================================

def run_pipeline(input_csv, output_csv):
    print(f"🚀 Starting Pipeline using input: {input_csv}")
    
    # 1. Load Data
    try:
        df = pd.read_csv(input_csv)
        print(f"   Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # 2. Prepare Storage for Results
    results = []

    # 3. Main Loop
    for index, row in df.iterrows():
        # Safety check: Clean up the book name text
        b_name = str(row['book_name']).strip().lower()
        claim_text = str(row['content']) # Assuming 'content' is the claim column
        row_id = row['id']

        print(f"   Processing ID {row_id}...")

        # --- STEP A: CALL MEMBER 1 (INGEST) ---
        # "Hey M1, give me the context for this book and this claim."
        context_result = get_context_from_ingest(b_name, claim_text)

        # --- STEP B: CALL MEMBER 3 (LOGIC) ---
        # "Hey M3, here is the context M1 gave me. Is the claim true?"
        label, rationale = verify_with_logic(context_result, claim_text)

        # 4. Store the Answer
        results.append({
            "id": row_id,
            "label": label,
            "rationale": rationale
        })

    # 5. Save Final Output
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"✅ Success! Results saved to {output_csv}")

# ==========================================
# PART 3: EXECUTION
# ==========================================
if __name__ == "__main__":
    # Define your file paths here
    INPUT_FILE = "data/mini_train.csv"  # Make sure this path matches your folder
    OUTPUT_FILE = "submission.csv"

    # Check if input exists to prevent crashing
    if os.path.exists(INPUT_FILE):
        run_pipeline(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"❌ Could not find {INPUT_FILE}. Please check your 'data' folder.")