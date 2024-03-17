import os
from flask import Flask, render_template, request, jsonify
from flask_assets import Environment, Bundle
from lib.chain import ChatChain

app = Flask(__name__)
assets = Environment(app)

scss = Bundle('styles.scss', filters='scss', output='gen/main.css')
assets.register('scss_all', scss)

chat_chain = ChatChain()

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        prompt = request.form("prompt")
        message = chat_chain.get_chat_chain().respond(prompt)
        print(message)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
