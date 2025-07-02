from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"

print("🔄 טוען את המודל...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("✅ מודל נטען בהצלחה!")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "Missing 'message' field."}), 400

    prompt = f"User: {user_input}\nAssistant:"
    response = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]['generated_text']
    answer = response.split("Assistant:")[-1].strip()
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
