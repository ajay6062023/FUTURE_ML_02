from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load models
category_model = joblib.load("category_model.pkl")
priority_model = joblib.load("priority_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def index():

    category = None
    priority = None

    if request.method == "POST":

        ticket = request.form["ticket"]

        ticket_vector = vectorizer.transform([ticket])

        category = category_model.predict(ticket_vector)[0]
        priority = priority_model.predict(ticket_vector)[0]

    return render_template("index.html",
                           category=category,
                           priority=priority)


if __name__ == "__main__":
    app.run(debug=True)