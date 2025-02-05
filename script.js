// Function to send a message to the chatbot and get a response
function sendChat() {
    const message = document.getElementById('chatInput').value;
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('chatOutput').innerText = "Chatbot says: " + data.response;
    });
}

// Function to fetch the maintenance prediction from the backend
function fetchMaintenancePrediction() {
    fetch('/api/maintenance-prediction')
        .then(response => response.json())
        .then(data => {
            document.getElementById('predictionResult').innerText = "Prediction: " + data.prediction;
        });
}

