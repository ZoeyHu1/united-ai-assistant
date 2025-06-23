# united-ai-assistant

**United Airlines – Multilingual Multi-Agent AI Assistant**  
🏆 Hackathon Project | Real-time AI Chatbot with Language Detection, Intent Classification, and Agent Routing


## 🚀 Overview

This is a **Streamlit-based multilingual airline assistant** built with LLMs and modular backend agents. It supports real-time conversations across languages and intelligently routes queries to topic-specific agents.

> Built for speed, modularity, and a better traveler experience.


## 🔑 Key Features

- 🌐 **Language Detection** (English, 中文, etc.)
- 🎯 **Intent Classification** (Flight, FAQ, Loyalty, Rec)
- 🤖 **Modular Agent Routing**  
  - `faq_agent.py`: Answers policy and common questions  
  - `flight_details_agent.py`: Handles flight features/status  
  - `recommendations_agent.py`: Gives hotel & travel suggestions  
  - `loyalty_program_agent.py`: Manages user mileage  
- 📊 **Real-Time Analytics** (query count, intent/language distribution)
- 💬 **Clean Streamlit UI** with message indicators and side config


## 🗂️ File Structure

```
├── chatbot/
│   ├── Processed_Hotels.csv
│   ├── faq_agent.py
│   ├── flight_details_agent.py
│   ├── loyalty_program_agent.py
│   ├── recommendations_agent.py
├── data/
│   └── Generated_data_notebook/
│       ├── flight_details_data_agent.ipynb
│       └── flight_recommendation data&process.ipynb
├── app.py                 # Streamlit frontend
├── README.md
```


## ⚙️ How to Run

1. Clone the repository  
2. Install requirements  
   ```bash
   pip install -r requirements.txt
3. Start the Streamlit app
   ```bash
   streamlit run app.py

## 🧠 How It Works

- The UI accepts user input  
- Classifier detects **language** and **intent**  
- Message is routed to the appropriate **agent module**  
- Agent response is returned and shown in the chat  
- Analytics update in real-time based on usage


## 🧪 Example Queries

- “What is your baggage policy?”  
- “Does flight UA0892 have WiFi?”  
- “How many miles do I have?”  
- “What hotels do you recommend in New York?”  
- “请问你们的行李政策是什么？”


## 📊 Analytics Dashboard

**Track:**

- Total queries  
- Languages used  
- Intent categories  
- Agent involvement  
- Session duration  


## 🛡️ Notes

Only partial functionality is available without a valid API key.

This chatbot is a **prototype**—production deployment should include:

- Error handling  
- Secure authentication  
- API rate limiting  
- Input sanitization  


