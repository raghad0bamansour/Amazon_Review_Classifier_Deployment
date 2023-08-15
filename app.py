from flask import Flask, render_template, request, jsonify
import pickle

cv = pickle.load(open("models/cv.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    review_text = request.form.get("content")
    tokenized_review = cv.transform([review_text])
    prediction = clf.predict(tokenized_review)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html",prediction=prediction, review_text=review_text)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)  # Get data posted as a json
    review_text = data['content']
    tokenized_review = cv.transform([review_text]) # X 
    prediction = clf.predict(tokenized_review)
    # If the email is spam prediction should be 1
    prediction = 1 if prediction == 1 else -1
    return jsonify({'prediction': prediction, 'review_text': review_text})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)