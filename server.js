require("dotenv").config();
const express = require("express");
const axios = require("axios");
const cors = require("cors");

const app = express();
app.use(express.json());  // Make sure Express can handle JSON
app.use(cors());

const GROQ_API_KEY = process.env.GROQ_API_KEY; 

// ✅ Make sure this endpoint exists
app.post("/chatbot", async (req, res) => {
    try {
        const userMessage = req.body.message;

        const response = await axios.post(
            "https://api.groq.com/openai/v1/chat/completions",
            {
                model: "groq-llama3-8b",
                messages: [{ role: "user", content: userMessage }],
            },
            {
                headers: { Authorization: `Bearer ${GROQ_API_KEY}` },
            }
        );

        console.log("Groq API Response:", response.data); // Debugging log
        res.json({ reply: response.data.choices[0].message.content });
    } catch (error) {
        console.error("Error:", error.response?.data || error.message);
        res.status(500).json({ error: "Error connecting to Groq API" });
    }
});

// ✅ Ensure your server listens on the correct port
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
