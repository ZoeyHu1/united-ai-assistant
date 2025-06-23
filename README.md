# 🤖 Multilingual Multi-Agent Airline Chatbot

A comprehensive AI-powered customer service solution that provides multilingual support for airline operations through specialized agent routing and real-time translation.

## Quick Start

- Install dependencies: `pip install -r requirements.txt`  
- Add your API keys to the environment variables  
- Run Streamlit UI: `streamlit run app.py`

## Required File Structure

```
airline-chatbot/
├── main.py                      # Main orchestrator & CLI interface
├── app.py                       # Streamlit web UI
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── agents/
│   ├── translator_agent.py      # Language detection & translation
│   ├── dispatcher_agent.py      # Intent classification & routing
│   ├── loyalty_program_agent.py # Feature 1: Loyalty program queries
│   ├── faq_agent.py             # Feature 2: FAQ answering
│   ├── recommendations_agent.py # Feature 3: Flight/hotel recommendations
│   ├── flight_disruption_agent.py # Feature 5: Alternative flights
│   └── flight_details_agent.py  # Feature 6: Detailed flight info
├── data/
│   ├── flight_features.csv      # Flight data for details agent
│   ├── Mock_flights.csv         # Flight search data
│   ├── Processed_Hotels.csv     # Hotel recommendations data
│   └── faq_data.json            # FAQ knowledge base
└── assets/
    └── united_logo.png          # UI logo (optional)
```

## Environment Variables

```
OPENAI_API_KEY=your_openai_key_here  
GROQ_API_KEY=your_groq_key_here  
GOOGLE_CLOUD_PROJECT_ID=your_project_id  # Optional for translation
```

## Features

- Multi-language support with auto-detection  
- Intent classification using LLM routing  
- 6 specialized agents for different airline services  
- Streamlit web interface with real-time chat  
- Analytics dashboard with usage metrics

## Core Agents

- **Loyalty Program** – MileagePlus queries, status, benefits  
- **FAQ** – Policies, baggage, check-in procedures  
- **Recommendations** – Flight/hotel suggestions  
- **Flight Details** – Amenities, WiFi, meals, seating  
- **Flight Disruption** – Rebooking, alternatives, changes  
- **General** – Fallback for unclassified queries
