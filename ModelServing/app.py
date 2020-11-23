import io
import json

from transformers.pipelines import pipeline
from flask import Flask, jsonify, request


app = Flask(__name__)
# imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
model_name = "deepset/electra-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

"""
QA_input = {
    'question': 'Why is model conversion important?', 'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
res = nlp(QA_input)
"""
def get_prediction(QA_input):
    res = nlp(QA_input)
    return res


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        QA_input = json.loads(request.data)
        answer = get_prediction(QA_input)
        return jsonify({'answer': answer['answer']})

# curl --header "Content-Type: application/json" --request POST --data '{​​​​​"question":"What is your name","context":"Hello, my name is Rudy"}​​​​​' http://localhost:5000/predict


if __name__ == '__main__':
    app.run()
