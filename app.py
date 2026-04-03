from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

model_path = "fake_job_model"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

model.eval()

@app.route("/", methods=["GET","POST"])
def home():

    result = None
    message = ""

    if request.method == "POST":

        message = request.form["message"]

        inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        if prediction == 1:
            result = "FAKE"
        else:
            result = "REAL"

    return render_template("index.html", result=result, message=message)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)