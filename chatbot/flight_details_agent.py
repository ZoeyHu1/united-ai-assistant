from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
import pandas as pd
import os
import re

class FlightDetailLookup:
    """Custom class to handle flight data lookup operations"""
    
    def __init__(self, csv_path: str = 'flight_features.csv'):
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.df)} flight records from {csv_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find flight data file: {csv_path}")
        except Exception as e:
            raise Exception(f"Error loading flight data: {str(e)}")
    
    def lookup_flight(self, flight_number: str) -> str:
        """Look up flight details by flight number"""
        flight_number = flight_number.strip().upper()
        
        if not flight_number.startswith('UA'):
            if flight_number.isdigit():
                flight_number = f"UA{flight_number.zfill(4)}" 
            else:
                return f"Invalid flight number format: {flight_number}. Please use format like 'UA892' or '892'"
        
        # Search for flight
        flight_data = self.df[self.df['flight_number'] == flight_number]
        
        if flight_data.empty:
            return f"Sorry, I couldn't find any information for flight {flight_number}. Please check the flight number and try again."

        # Extract flight details
        flight = flight_data.iloc[0]
        
        # Format the response
        response = f"""
**United Airlines Flight {flight['flight_number']}**

**Aircraft & Seating:**
   • Aircraft: {flight['aircraft_type']}
   • Seat Configuration: {flight['seat_config']}
   • Total Seats: {flight['total_seats']}
   • Exit Row Seats: {flight['num_of_exit_row_seats']}

**Connectivity & Power:**
   • WiFi: {'Available' if flight['wifi'] else 'Not Available'}
   • WiFi Pricing: {flight['wifi_price_range']}
   • USB Charging: {'Available' if flight['usb'] else 'Not Available'}
   • Power Outlets: {'Available' if flight['power_outlets'] else 'Not Available'}
   • Entertainment: {flight['entertainment']}

**Dining & Service:**
   • Route Type: {flight['route_type']}
   • Meal Service: {flight['meal_type']}

**Travel Info:**
   • Baggage Policy: {flight['baggage_policy']}
   • Boarding/Lounge: {flight['lounge_access']}

**Important Notes:**
   {flight['notes']}
        """.strip()
        
        return response

class QueryClassifier:
    """Classify queries to determine appropriate response templates"""
    
    @staticmethod
    def classify_query(query: str) -> str:
        """
        Classify the type of query to use appropriate response template
        
        Returns:
            Query type: 'meal', 'wifi', 'seating', 'entertainment', 'comparison', 'general'
        """
        query_lower = query.lower()
        
        # Meal-related queries
        meal_keywords = ['meal', 'food', 'vegetarian', 'vegan', 'kosher', 'halal', 'diet', 'dining', 'eat']
        if any(keyword in query_lower for keyword in meal_keywords):
            return 'meal'
        
        # WiFi/connectivity queries  
        wifi_keywords = ['wifi', 'wi-fi', 'internet', 'connect', 'online']
        if any(keyword in query_lower for keyword in wifi_keywords):
            return 'wifi'
        
        # Seating queries
        seat_keywords = ['seat', 'seating', 'configuration', 'exit row', 'legroom', 'window', 'aisle']
        if any(keyword in query_lower for keyword in seat_keywords):
            return 'seating'
        
        # Entertainment queries
        entertainment_keywords = ['entertainment', 'movie', 'tv', 'screen', 'streaming', 'games']
        if any(keyword in query_lower for keyword in entertainment_keywords):
            return 'entertainment'
        
        # Power/charging queries
        power_keywords = ['usb', 'charging', 'power', 'outlet', 'plug', 'charge', 'battery']
        if any(keyword in query_lower for keyword in power_keywords):
            return 'power'
        
        # Comparison queries
        comparison_keywords = ['compare', 'better', 'difference', 'vs', 'versus', 'which']
        if any(keyword in query_lower for keyword in comparison_keywords):
            return 'comparison'
        
        return 'general'

class UnitedFlightAgent:
    """Enhanced agent with smart prompt templates for different query types"""
    
    def __init__(self, csv_path: str = 'flight_features.csv', groq_api_key: str = None, memory_window: int = 10):
        """Initialize the Enhanced United Flight Agent with Memory and Templates"""
        
        # Initialize flight lookup
        self.flight_lookup = FlightDetailLookup(csv_path)
        self.query_classifier = QueryClassifier()
        
        # Create the flight detail tool
        self.flight_tool = Tool(
            name="flight_detail_lookup",
            description="""
            Use this tool to look up detailed information about United Airlines flights.
            Input should be a flight number (e.g., 'UA892', 'UA123', or just '892').
            
            This tool provides comprehensive flight information including:
            - Aircraft type and seating configuration  
            - WiFi availability and pricing
            - Meal service details (vegetarian, standard, etc.)
            - USB charging and power outlets
            - Entertainment options
            - Baggage policies
            - Special notes and requirements
            
            Always use this tool when users ask about specific flight features, amenities, or services.
            """,
            func=self.flight_lookup.lookup_flight
        )
        
        # Set up prompt templates for different query types
        self._setup_prompt_templates()
        
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Initialize Groq Llama 3 model
        try:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                model="llama3-70b-8192",
                temperature=0,
                max_tokens=4096,
                timeout=30,
                max_retries=2
            )
         
        except ImportError as e:
            raise ImportError(
                "langchain-groq not installed!\n"
                "Please install it with: pip install langchain-groq\n"
                f"Original error: {str(e)}"
            )
        except Exception as e:
            raise Exception(f"Error initializing Groq LLM: {str(e)}")
        
        # MEMORY SETUP 
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        # Initialize the agent WITH memory
        self.agent = initialize_agent(
            tools=[self.flight_tool],
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
            memory=self.memory,  
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
        print("🤖 Enhanced United Airlines Flight Agent with Smart Templates initialized!")
    
    def _setup_prompt_templates(self):
        """Set up specialized prompt templates for different query types"""
        
        self.templates = {
            'meal': """
You are a helpful United Airlines customer service agent specializing in meal and dining services. 
When answering about meals:

1. Always mention if the specific meal type is available (vegetarian, kosher, halal, etc.)
2. Provide details about how to request special meals (24-48 hours advance notice)
3. Explain what's typically included in the meal service for that route type
4. Mention any additional food options (buy-on-board, snacks, beverages)
5. Note any restrictions or special requirements

Be helpful and proactive in your meal-related advice.

Customer question: {question}
""",

            'wifi': """
You are a helpful United Airlines customer service agent specializing in connectivity and WiFi services.
When answering about WiFi:

1. Clearly state if WiFi is available on the specific flight
2. Provide exact pricing information if available
3. Explain the coverage area (domestic vs international limitations)
4. Mention alternative connectivity options if WiFi isn't available
5. Note any complimentary WiFi benefits for certain passengers

Be specific about pricing, coverage, and practical usage information.

Customer question: {question}
""",

            'seating': """
You are a helpful United Airlines customer service agent specializing in seating and aircraft configurations.
When answering about seating:

1. Describe the exact seat configuration (3-3, 2-4-2, etc.)
2. Explain the total number of seats and aircraft type
3. Detail exit row availability and restrictions
4. Mention seat selection options and any associated fees
5. Suggest the best seat options for different passenger needs

Provide practical seating advice and clear configuration details.

Customer question: {question}
""",

            'entertainment': """
You are a helpful United Airlines customer service agent specializing in in-flight entertainment.
When answering about entertainment:

1. Describe the specific entertainment system available (seatback screens vs streaming)
2. List types of content available (movies, TV shows, music, games)
3. Explain how to access the entertainment (app requirements, device compatibility)
4. Mention any complimentary vs paid content
5. Note any restrictions or requirements

Give comprehensive entertainment guidance for the specific flight.

Customer question: {question}
""",

            'power': """
You are a helpful United Airlines customer service agent specializing in power and charging options.
When answering about power/charging:

1. Clearly state USB charging availability at seats
2. Specify power outlet types and locations if available
3. Mention any device compatibility considerations
4. Note any restrictions on device usage during flight phases

Provide practical charging advice and clear technical details.

Customer question: {question}
""",

            'comparison': """
You are a helpful United Airlines customer service agent specializing in flight comparisons.
When comparing flights:

1. Use the flight lookup tool for each flight mentioned
2. Create a clear side-by-side comparison of requested features
3. Highlight the key differences that matter most to travelers
4. Provide specific recommendations based on passenger priorities
5. Mention any additional considerations (route type, aircraft differences)
6. Suggest factors to help make the decision

Give a comprehensive, objective comparison with clear recommendations.

Customer question: {question}
""",

            'general': """
You are a helpful United Airlines customer service agent providing comprehensive flight information.
When answering general questions:

1. Use the flight lookup tool to get specific flight details
2. Provide complete, accurate information from the flight data
3. Anticipate follow-up questions and provide proactive information
4. Maintain a friendly, professional, and helpful tone
5. Reference specific flight details when available

Always strive to be more helpful than the customer expects.

Customer question: {question}
"""
        }
    
    def query(self, question: str) -> str:
        """
        Process a user question with smart template selection
        
        Args:
            question: User's question about flight details
            
        Returns:
            Enhanced agent response with appropriate template context
        """
        try:
            query_type = self.query_classifier.classify_query(question)
            template = self.templates.get(query_type, self.templates['general'])
            enhanced_question = template.format(question=question)
            response = self.agent.run(enhanced_question)
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def get_memory(self) -> str:
        """View current conversation memory"""
        if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
            memory_content = []
            for i, msg in enumerate(self.memory.chat_memory.messages):
                if hasattr(msg, 'content'):
                    role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Agent"
                    content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                    memory_content.append(f"{role}: {content}")
            return "\n\n".join(memory_content)
        return "No conversation history yet."

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        print("Memory cleared! Starting fresh conversation.")

    def get_memory_summary(self) -> dict:
        """Get a summary of memory usage"""
        if hasattr(self.memory, 'chat_memory'):
            total_messages = len(self.memory.chat_memory.messages)
            return {
                "total_messages": total_messages,
                "memory_window": self.memory.k,
                "memory_usage": f"{total_messages}/{self.memory.k * 2}", 
                "memory_full": total_messages >= (self.memory.k * 2)
            }
        return {"total_messages": 0, "memory_window": self.memory.k, "memory_usage": "0/0"}
    
    def test_query_classification(self, test_queries: list = None) -> dict:
        """Test the query classification system"""
        if test_queries is None:
            test_queries = [
                "Does flight UA0892 have vegetarian meals?",
                "Is WiFi available on UA0123?",
                "What's the seat configuration for UA0456?",
                "Does UA0789 have entertainment screens?",
                "Can I charge my phone on this flight?",
                "Compare WiFi between UA0111 and UA0222",
                "Tell me about flight UA0333"
            ]
        
        results = {}
        for query in test_queries:
            query_type = self.query_classifier.classify_query(query)
            results[query] = query_type
        
        return results

if __name__ == "__main__":
    # Demonstrate query classification
    print("Query Classification Examples:")
    agent = UnitedFlightAgent.__new__(UnitedFlightAgent)  
    agent.query_classifier = QueryClassifier()
    
    test_results = agent.test_query_classification()
    for query, query_type in test_results.items():
        print(f"'{query}' → {query_type}")