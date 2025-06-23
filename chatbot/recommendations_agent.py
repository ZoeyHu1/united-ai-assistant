import requests
import json
import pandas as pd
import time
from datetime import datetime
from random import sample
from urllib.parse import quote

class RecommendationsAgent:
    def __init__(self,
                 api_key: str,
                 api_url: str,
                 model: str,
                 flights_csv: str,
                 hotels_csv: str):
        # HTTP headers for LLM API
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.api_url     = api_url
        self.model       = model
        self.flights_csv = flights_csv

        # load pre-processed hotel data
        # expects columns: hotel_name, hotel_brand, hotel_grade, hotel_link
        self.hotels_df = pd.read_csv(hotels_csv)

        # system prompt for flight extraction
        self.system_prompt = """
You are a flight booking assistant. When given a user's input, you must output a JSON object with these fields:

Required:
- departure_city
- arrival_city
- departure_date

Optional (use null if not provided):
- departure_time
- arrival_time
- max_price
- wifi_available
- direct

Normalize any free-form departure_date into strict "YYYY-MM-DD" format.
Always respond with JSON only, filling any missing fields with null.
"""

    def extract_initial(self, user_msg: str) -> dict:
        """Call LLM to parse user's free-form request into structured JSON."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_msg}
            ]
        }
        r = requests.post(self.api_url, headers=self.headers, json=payload)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"].strip()
        return json.loads(txt)

    def filter_flights(self, criteria: dict) -> pd.DataFrame:
        """Filter the mock flights CSV according to the user's criteria."""
        df = pd.read_csv(self.flights_csv)
        df['departure_time'] = pd.to_datetime(df['departure_time'])
        df['arrival_time']   = pd.to_datetime(df['arrival_time'])

        # required filters
        filtered = df[
            (df['departure_city'] == criteria['departure_city']) &
            (df['arrival_city']   == criteria['arrival_city']) &
            (df['departure_time'].dt.date.astype(str) == criteria['departure_date'])
        ]

        # optional filters
        if criteria.get('departure_time'):
            t0 = datetime.strptime(criteria['departure_time'], "%H:%M").time()
            filtered = filtered[filtered['departure_time'].dt.time == t0]
        if criteria.get('arrival_time'):
            t1 = datetime.strptime(criteria['arrival_time'], "%H:%M").time()
            filtered = filtered[filtered['arrival_time'].dt.time == t1]
        if criteria.get('max_price') is not None:
            filtered = filtered[filtered['price_usd'] <= criteria['max_price']]
        if criteria.get('wifi_available') is not None:
            filtered = filtered[filtered['wifi_available'] == criteria['wifi_available']]
        if criteria.get('direct') is not None:
            filtered = filtered[filtered['direct'] == criteria['direct']]

        return filtered

    def recommend_hotels(self, arrival_city: str):
        """
        Ask the user if they want hotel recommendations.
        If yes, ask preference (brand group or star grade) and show 3–5 picks.
        """
        ans = input(f"\nWould you like hotel recommendations in {arrival_city}? (yes/no)\n> ").strip().lower()
        if ans not in ("yes", "y"):
            print("Okay, happy travels! Let me know if I can help with anything else.")
            return

        # choose filter dimension
        choice = input("Filter hotels by Brand group or Star grade? (brand/grade)\n> ").strip().lower()
        if choice.startswith("b"):
            groups = sorted(self.hotels_df['hotel_brand'].unique())
            pref = input(f"Available brand groups: {', '.join(groups)}\nEnter preferred brand group:\n> ").strip()
            candidates = self.hotels_df[self.hotels_df['hotel_brand'] == pref]
        else:
            grades = sorted(self.hotels_df['hotel_grade'].unique())
            pref = input(f"Available grades: {', '.join(grades)}\nEnter preferred star grade:\n> ").strip()
            candidates = self.hotels_df[self.hotels_df['hotel_grade'] == pref]

        if candidates.empty:
            print(
                " No hotels found for that category, but you could go through this link "
                " to find more hotel options: "
                " https://www.united.com/en/us/fly/mileageplus/partners/hotels.html")
            return

        # randomly sample up to 5
        picks = candidates.sample(min(5, len(candidates)))
        print(f"\n 🏨 Here are your hotel recommendations in {arrival_city}:")
        print(f"\n 🎁 Great News! Book with these hotels could start earning rewards. You can earn MileagePlus award miles for your hotel.")
        for _, row in picks.iterrows():
            print(f"- {row['hotel_name']} ({row['hotel_brand']}, {row['hotel_grade']})")
            print(f"  Link: {row['hotel_link']}\n")

    def run(self):
        """Main interaction loop."""
        required = ["departure_city", "arrival_city", "departure_date"]
        optional = ["departure_time", "arrival_time", "max_price", "wifi_available", "direct"]

        prompts_req = {
            "departure_city": "Please tell me your departure city:",
            "arrival_city":   "Please tell me your arrival city:",
            "departure_date": "Please tell me your departure date (YYYY-MM-DD or free-form):"
        }
        prompts_opt = {
            "departure_time": "Preferred departure time? (HH:MM or null):",
            "arrival_time":   "Preferred arrival time? (HH:MM or null):",
            "max_price":      "Maximum price? (e.g. 300 or $300, or null):",
            "wifi_available": "Need in-flight WiFi? (yes/no/null):",
            "direct":         "Require a direct flight? (yes/no/null):"
        }

        # 1) Extract initial criteria
        user_input = input('Enter request (e.g. "Book flight from A to B on July 4th"):\n> ')
        context    = self.extract_initial(user_input)

        # 2) Fill missing required
        for f in required:
            while not context.get(f):
                val = input(prompts_req[f] + "\n> ").strip()
                context[f] = val or None

        # 3) Fill missing optional
        for f in optional:
            if f not in context or context[f] is None:
                val = input(prompts_opt[f] + "\n> ").strip()
                if f == "max_price":
                    if val.lower() in ("", "null"):
                        context[f] = None
                    else:
                        try:
                            context[f] = float(val.replace("$","").replace(",",""))
                        except:
                            context[f] = None
                elif f in ("wifi_available", "direct"):
                    v = val.lower()
                    if v in ("yes","y","true","t"):
                        context[f] = True
                    elif v in ("no","n","false","f"):
                        context[f] = False
                    else:
                        context[f] = None
                else:
                    context[f] = None if val.lower() in ("","null") else val

        # 4) Confirm & filter flights
        while True:
            print("\nHere is your complete flight request:")
            print(json.dumps(context, indent=2))
            yn = input("Is this correct? (yes/no)\n> ").strip().lower()
            if yn in ("yes","y"):
                print("\n✅ Confirmed. Recommending flights that best suit you...\n")
                matches = self.filter_flights(context)

                if matches.empty:
                    print(
                " No flights match your criteria, but you could go through this link "
                " to find more flight options: "
                " https://www.united.com/en/us/book-flight/united-reservations")
                else:
                    print(matches.to_string(index=False))

                    # Build United booking link
                    dep_code = (matches['departure_airport'].iloc[0]
                                if 'departure_airport' in matches else
                                context['departure_city'][:3].upper())
                    dest_str = context['arrival_city']
                    params = [
                        ("f", dep_code), ("t", dest_str),
                        ("d", context['departure_date']),
                        ("tt","1"), ("st","bestmatches"), ("sc","7"),
                        ("px","1"), ("taxng","1"), ("newHP","True"),
                        ("clm","7"), ("tqp","R")
                    ]
                    qs = "&".join(
                        f"{k}={quote(v)}" if k=="t" else f"{k}={v}"
                        for k,v in params
                    )
                    link = f"https://www.united.com/en/us/fsr/choose-flights?{qs}"
                    print("\n ✈️ Here is the booking link for the flight that suits you — you can click it to go directly to the booking page ：\n", link)

                # 5) Hotel recommendation step
                self.recommend_hotels(context['arrival_city'])
                
            
  
                # 6) Car rental recommendation
                car_ans = input("\nWould you like a car rental recommendation? (yes/no)\n> ").strip().lower()
                if car_ans in ("yes","y","of course"):
                    print("\n 🚗 Car rental link:\n"
                          "https://cars.united.com/?clientid=569554"
                          "&utm_source=loyalty&utm_medium=uacom"
                          "&utm_campaign=transportpartnerspage"
                          "&utm_content=car_earn#/searchcars\n")
                    print(
                        "Book a car with this link and earn up to 1,250 award miles per qualifying rental. "
                        "You can also save up to 35% with the MileagePlus® Avis discount3. "
                        "If you're a Premier® member or United Cardmember, you can also get Avis elite status for free."
                    )

                # now exit
                return

            elif yn in ("no","n"):
                fld = input("Which field do you want to update?\n> ").strip()
                if fld not in required + optional:
                    print("Unknown field.")
                    continue
                new = input(f"Enter new value for {fld} (or 'null'):\n> ").strip()
                if fld == "max_price":
                    try:
                        context[fld] = float(new.replace("$","").replace(",","")) if new.lower() not in ("","null") else None
                    except:
                        context[fld] = None
                elif fld in ("wifi_available","direct"):
                    lv = new.lower()
                    context[fld] = True if lv in ("yes","y","true") else False if lv in ("no","n","false") else None
                else:
                    context[fld] = None if new.lower() in ("","null") else new
            else:
                print("Please answer yes or no.")

if __name__ == "__main__":
    # === Configuration ===
    API_KEY     = "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"
    API_URL     = "https://api.groq.com/openai/v1/chat/completions"
    MODEL       = "llama3-70b-8192"
    FLIGHTS_CSV = "Mock_flights.csv"
    HOTELS_CSV  = "Processed_Hotels.csv"

    ra = RecommendationsAgent(
        api_key    = API_KEY,
        api_url    = API_URL,
        model      = MODEL,
        flights_csv= FLIGHTS_CSV,
        hotels_csv = HOTELS_CSV
    )
    ra.run()
