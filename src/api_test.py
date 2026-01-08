# src/api_test.py
from openai import OpenAI

# 1. Point to your local Ollama server
# (Ollama runs on port 11434 by default)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # Required parameter, but Ollama ignores the value
)

try:
    print("Attempting to connect to Ollama...")

    # 2. Send the handshake message
    # IMPORTANT: Change "llama3" to "mistral" or "gemma" if that is what you have installed.
    response = client.chat.completions.create(
        model="mistral", 
        messages=[
            {"role": "user", "content": "Hello! Are you ready for the KDSH hackathon?"}
        ]
    )

    # 3. Print the success message
    print("\nSuccess! Ollama replied:")
    print(f"--> {response.choices[0].message.content}")

except Exception as e:
    print(f"\nConnection Failed. Error: {e}")
    print("\nTroubleshooting Tip: Is Ollama running? Run 'ollama serve' in a separate terminal.")