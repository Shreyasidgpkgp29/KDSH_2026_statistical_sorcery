# src/test_mistral.py
import pathway as pw
from openai import OpenAI
import pandas as pd

print("--- PHASE 1: TESTING PATHWAY ---")
# Create a tiny table to prove Pathway is working
try:
    df = pd.DataFrame({"claim": ["Mistral is fast"], "value": [100]})
    table = pw.debug.table_from_pandas(df)
    print("✅ Pathway is installed and working!")
except Exception as e:
    print(f"❌ Pathway Error: {e}")

print("\n--- PHASE 2: TESTING OLLAMA (MISTRAL) ---")
try:
    # Connect to local Ollama
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama', 
    )
    
    print("Sending message to Mistral...")
    response = client.chat.completions.create(
        model="mistral",  # We are explicitly using Mistral
        messages=[{"role": "user", "content": "Reply with exactly three words: System All Green."}]
    )
    
    print(f"✅ Mistral Replied: {response.choices[0].message.content}")

except Exception as e:
    print(f"❌ Ollama Error: {e}")
    print("👉 Did you forget to run 'ollama serve' in another terminal?")
    print("👉 Did you run 'ollama pull mistral'?")