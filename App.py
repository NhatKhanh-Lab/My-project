from flask import Flask, render_template, request
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Hàm tải mô hình và vectorizer
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer

# Hàm xử lý và dự đoán
def predict(model, vectorizer, title, text, subject, date):
    input_data = [title + ' ' + text + ' ' + subject + ' ' + date]
    input_vector = vectorizer.transform(input_data)
    prediction = model.predict(input_vector)

    return "Tin thật" if prediction == 1 else "Tin giả"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        title = request.form["title"]
        text = request.form["text"]
        subject = request.form["subject"]
        date = request.form["date"]

        # Kiểm tra nếu thông tin người dùng nhập vào đầy đủ
        if not all([title, text, subject, date]):
            result = "Vui lòng nhập đầy đủ thông tin vào tất cả các ô."
        else:
            # Load mô hình và vectorizer
            model, vectorizer = load_model()
            
            # Dự đoán
            result = predict(model, vectorizer, title, text, subject, date)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
