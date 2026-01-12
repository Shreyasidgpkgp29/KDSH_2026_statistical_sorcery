import json
import re
from langchain_community.llms import Ollama

llm = Ollama(model="mistral", temperature=0)

def verify_claim(context, claim):
    prompt = f"""
You are a Logic Consistency Engine. 
Your goal is to determine if a specific CLAIM is compatible with a provided CONTEXT.

THE "MECHANISM" LOGIC RULE (CRITICAL):
1.  **Context = Observation:** Treat the Context as a camera recording what was *seen* and *heard*.
2.  **Claim = Mechanism:** Treat the Claim as the "Behind the Scenes" explanation.
3.  **The Compatibility Check:**
    - If the Context says "She cleared her throat and her voice boomed," and the Claim says "She is mute and used a voice double," **THIS IS VALID (1).**
    - Reasoning: The "Mechanism" (Voice Double) explains the "Observation" (Booming Sound). The "throat clearing" is interpreted as pantomime/acting to sell the illusion.
    - **Explicit Mechanisms Only:** You may only output 1 if the CLAIM explicitly provides the method (e.g., "acting," "recording," "double") that explains the CONTEXT.
    - **No Hallucinated Justifications:** Do NOT invent a "trick" or "illusion" if the CLAIM does not mention one. If the CLAIM is a simple statement that contradicts the CONTEXT without explaining how (e.g., Claim: "He is dead" vs. Context: "He is running a marathon"), output 0.
    - VALID (1): Context: "She spoke." | Claim: "She is mute but used a hidden speaker." (The claim explains the observation).
    - INVALID (0): Context: "She spoke." | Claim: "She is mute." (The claim contradicts the observation and offers no mechanism).

YOUR TASK:
1. Analyze if the CLAIM (Mechanism) makes the CONTEXT (Observation) *physically impossible* to have occurred.
2. If the Claim explains *how* the Context happened (even via a trick), output 1.

OUTPUT FORMAT (STRICT):
Do not print "```json" or any conversational text. 
Only print the JSON object followed immediately by the number on a new line.

CASE 1: If Consistent (Result 1):
{{
"explanation" : "One sentence explaining how the Mechanism explains the Observation."
}}
1

CASE 2: If Contradiction (Result 0):
{{
"target" : "The part of the claim that fails",
"proof" : "The exact excerpt from Context",
"explanation" : "Why the mechanism fails"
}}
0

CONTEXT: ```{context}```
CLAIM: ```{claim}```
""" 
    try:
        response = llm.invoke(prompt).strip()

        lines = response.splitlines()
        if not lines:
            return 1, "No response from LLM."

        label_match = re.search(r'[01]', lines[-1])
        label = int(label_match.group()) if label_match else 1

        json_str = "\n".join(lines[:-1]).strip()
        if "```" in json_str:
            json_str = re.sub(r'^```(?:json)?\n?|```$', '', json_str, flags=re.MULTILINE).strip()
    
        data = json.loads(json_str)
        
        if label == 0:
            rationale = f"Target: {data.get('target', 'N/A')} | Proof: {data.get('proof', 'N/A')} | Why: {data.get('explanation', 'N/A')}"
        else:
            rationale = data.get("explanation", "Consistent with context.")
    
        return label, rationale

    except Exception as e:
        return 1, f"Parsing error: {str(e)}"

if __name__ == "__main__":
    test_context = "The man was seen running through the park at noon."
    test_claim = "The man has been paralyzed from the waist down since birth."
    label, rationale = verify_claim(test_context, test_claim)
    print(f"Prediction: {label}\nRationale: {rationale}")