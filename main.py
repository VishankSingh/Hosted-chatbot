from flask import Flask, request, render_template
from chatbot import chat

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('portal.html')

@app.route('/', methods=['POST'])
def func():
    text = request.form['in']

    return render_template('portal.html', text = chat(text))



if __name__ == "__main__":
    app.run()