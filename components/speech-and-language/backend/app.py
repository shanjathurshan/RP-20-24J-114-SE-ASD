from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Loading GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_similar_words(word):
    input_text = f"Suggest words that sound similar to: {word}. "
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generating output from the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.7)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    similar_words = generated_text.replace(input_text, '').strip().split(',')

    return [w.strip() for w in similar_words]

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    word = data.get("word", "")
    if not word:
        return jsonify({"error": "No word provided"}), 400
    similar_words = generate_similar_words(word)
    return jsonify({"similar_words": similar_words})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)