import json
from langchain_community.llms import Ollama

# Setup the Brain (lama)Ol
llm = Ollama(model="mistral", temperature=0)

def verify_claim(context, claim):
    """
    Member 3's core function.
    Takes context (from Member 1) and a claim (from the CSV).
    Returns: label (0 or 1), rationale (string).
    """
    
    prompt = f"""
    You are a pedantic literary auditor.
    CONTEXT: {context}
    CLAIM: {claim}

    TASK:
    1. Check if the CLAIM is contradicted by the CONTEXT.
    2. Lack of information is NOT a contradiction.
    3. Output ONLY a JSON object with these keys:
       "label": 0 (if contradicted) or 1 (if consistent)
       "rationale": "One sentence explaining the proof"

    JSON:
    """
    
    try:
        response = llm.invoke(prompt)
        # Parse the JSON output from the LLM
        data = json.loads(response)
        return data.get("label", 1), data.get("rationale", "No contradiction found.")
    except Exception as e:
        # Fallback if the LLM output isn't perfect JSON
        return 1, f"Error parsing response: {str(e)}"

if __name__ == "__main__":
    # Test block to verify it works independently
    test_context = "Sarah was a florist in Oakhaven."
    test_claim = "Sarah was a professional pilot."
    label, reason = verify_claim(test_context, test_claim)
    print(f"Label: {label}\nRationale: {reason}")