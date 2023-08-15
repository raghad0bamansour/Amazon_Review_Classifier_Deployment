from flask import Flask, render_template, request, jsonify
from utils import make_prediction
app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    review_text = request.form.get("content")
    prediction = make_prediction(review_text)
    return render_template("index.html",prediction=prediction, review_text=review_text)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)  # Get data posted as a json
    review_text = data['content']
    prediction = make_prediction(review_text)
    return jsonify({'prediction': prediction, 'review_text': review_text})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)