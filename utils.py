import pickle

cv = pickle.load(open("models/cv.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))

def make_prediction(review_text):
    tokenized_review = cv.transform([review_text])
    prediction = clf.predict(tokenized_review)
    #prediction = 1 if prediction == 1 else -1
    return prediction
