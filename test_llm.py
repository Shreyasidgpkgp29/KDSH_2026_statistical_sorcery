from openai import OpenAI

# Point to local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" # Required but unused
)

response = client.chat.completions.create(
    model="mistral", # Or "llama3" depending on what you pulled
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello Team KDSH' and nothing else."}
    ]
)

print("LLM Response:", response.choices[0].message.content)