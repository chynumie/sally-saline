from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import chatbot  # Import your chatbot logic

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    # Get response from your chatbot
    bot_response = chatbot(user_message)  # Modify your chatbot function to accept a message and return a response
    
    return jsonify({'reply': bot_response})

if __name__ == '__main__':
    app.run(debug=True) 