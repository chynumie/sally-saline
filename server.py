from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from chatbot import chatbot  # Import your chatbot logic
except Exception as e:
    logger.error(f"Failed to import chatbot: {str(e)}")
    logger.error(traceback.format_exc())

app = Flask(__name__)
CORS(app)

# Add root route for Render health checks
@app.route('/', methods=['GET'])
def root():
    return jsonify({'status': 'healthy'}), 200

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        # Get response from your chatbot
        bot_response = chatbot(user_message)
        
        return jsonify({'reply': bot_response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'reply': 'Sorry, an error occurred while processing your request.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    try:
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc()) 