import os
from flask import Flask, render_template, request, jsonify
from flask_assets import Environment, Bundle
from lib.chain import ChatChain

app = Flask(__name__)
assets = Environment(app)

scss = Bundle('styles.scss', filters='scss', output='gen/main.css')
assets.register('scss_all', scss)

chat_chain = ChatChain()
chat_history = []

@app.route('/', methods=['GET','POST'])
def index():
    processed_list = []
    if request.method == 'POST':
        prompt = request.form.get("prompt")
        answer = chat_chain.get_response(prompt)
        chat_history.append({'prompt': prompt})
        chat_history.append({'answer:': answer})
        processed_list = [(i + 1, item) for i, item in enumerate(chat_history)]
    return render_template('index.html', chat_history = processed_list)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
