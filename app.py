from flask import Flask, render_template, request
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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)