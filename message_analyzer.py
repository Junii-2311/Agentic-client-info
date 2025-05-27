import psycopg2
from urllib.parse import urlparse
import pandas as pd
from datetime import datetime
from textwrap import dedent
import google.generativeai as genai
from agno.agent import Agent
from agno.tools.exa import ExaTools
import dateutil.parser as dp




def normalize_ts(ts_str):
    if ts_str.lower() in ("none", ""):
        return "none"
    dt = dp.parse(ts_str)
    if dt.time() == datetime.min.time():
        return "none"
    return dt.isoformat()


# === CONFIGURATION ===
genai.configure(api_key="AIzaSyCARQ2mkQV3dd-TzWo79Q5wkloah1Aqiac")
EXA_API_KEY = "3a27a6bf-82e8-48f9-b0e8-fcb6bde6a8f6"
today = datetime.now().strftime("%Y-%m-%d")



# === DATABASE CONNECTION ===
def connect_to_db():
    """Connect to the PostgreSQL database using the provided connection string."""
    
    # Your provided connection string
    db_url = "postgres://readonly:p2e0dfd8702aac2f2b98b80f2e14430fb7d3cec6f8aec1770701283042b112712@ec2-23-20-93-193.compute-1.amazonaws.com:5432/d5pt3225ki095v"
    
    # Parse the database URL into its components
    result = urlparse(db_url)
    
    # Extract the connection details from the URL
    dbname = result.path[1:]  # Remove the leading '/'
    user = result.username
    password = result.password
    host = result.hostname
    port = result.port
    
    
    
    # Establish the connection
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    return conn




def fetch_qualified_clients(conn):
    """Fetch clients whose current_stage is 4 or greater."""
    query = """
    SELECT client_id 
    FROM client_stage_progression 
    WHERE current_stage >= 4;
    """
    cur = conn.cursor()
    cur.execute(query)
    clients = cur.fetchall()
    cur.close()
    return clients




# def fetch_client_messages(conn, client_id):
#     """Fetch messages for a specific client from client_fub_messages."""
#     query = """
#     SELECT message 
#     FROM client_fub_messages 
#     WHERE client_id = %s;
#     """
#     cur = conn.cursor()
#     cur.execute(query, (client_id,))
#     messages = cur.fetchall()
#     cur.close()
#     return [msg[0] for msg in messages]




def fetch_client_messages(conn, client_id):
    """
    Fetch messages for a specific client:
    - Try from 'public.textmessage' table (status in 'Sent', 'Received').
    - If not found, fall back to 'client_fub_messages' table.
    Returns a list of message strings (in order).
    """
    cur = conn.cursor()
    try:
        # 1. Try public.textmessage first
        query1 = """
        SELECT message
        FROM public.textmessage
        WHERE client_id = %s
          AND status IN ('Sent', 'Received')
        ORDER BY created ASC
        """
        cur.execute(query1, (client_id,))
        rows = cur.fetchall()
        if rows and len(rows) > 0 and rows[0][0] is not None:
            messages = [row[0] for row in rows if row[0] is not None]
            print(f"Fetched {len(messages)} messages from 'public.textmessage' for client {client_id}.")
            return messages

        # 2. If nothing found, try client_fub_messages
        print(f"No messages found in 'public.textmessage' for client {client_id}, checking 'client_fub_messages'...")
        query2 = """
        SELECT message
        FROM client_fub_messages
        WHERE client_id = %s
        """
        cur.execute(query2, (client_id,))
        rows2 = cur.fetchall()
        messages2 = [row[0] for row in rows2 if row[0] is not None]
        if messages2:
            print(f"Fetched {len(messages2)} messages from 'client_fub_messages' for client {client_id}.")
        else:
            print(f"No messages found for client {client_id} in either table.")
        return messages2

    except Exception as e:
        print(f"Error fetching client messages for client {client_id}: {e}")
        return []
    finally:
        cur.close()








# === GEMINI MODEL WRAPPER ===
class GeminiChat:
    assistant_message_role = "assistant"

    def __init__(self, id=None):
        self.id = id or 'gemini-2.0-flash'

    def response_stream(self, messages, **kwargs):
        if isinstance(messages, list):
            last = messages[-1]
            prompt = last.content if hasattr(last, 'content') else last.get('content', str(last))
        else:
            prompt = messages

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)

        class GeminiResponse:
            def __init__(self, content):
                self.event = "assistant_response"
                self.content = content

        yield GeminiResponse(response.text)

    def get_instructions_for_model(self, tools=None):
        return ""

    def get_system_message_for_model(self, tools=None):
        return ""


# === AGENT DEFINITION ===
message_analyzer = Agent(
    model=GeminiChat(),
    tools=[ExaTools(api_key=EXA_API_KEY, start_published_date=today, type="keyword")],
    description=dedent("""
        You manage a team of agents who collect client requirements.
        Your primary goal is to track all building-related interactions:
        - Buildings sent to clients
        - Client responses: acceptance, rejection (with reason), or replacement requests
        - Full history of interaction with timestamps and actions (tour scheduled, canceled, etc.)
    """),
    instructions=dedent("""
        From the provided chat messages, extract structured building-level data only.

        For each building mentioned, return an object with:
        - building_name
        - tour_time (if scheduled)
        - status: one of [sent, toured, rejected, cancelled, replaced]
        - actions: list of interaction types, e.g., ["Call", "Guest Card", "Website", "Email"]
        - booking_method: "Call" | "Email" | "Online" | "Not mentioned"
        - tour_type:
            - "In-Person Tour" if a group chat was created
            - "Virtual Tour" if no group chat was created
            - or "Self Guided Tour" / "Videos Only" if explicitly mentioned
        - notes: optional info like rejection reason, replacement info, or cancellation context
        - timestamp: ISO format of earliest mention

        ❌ Do not include:
        - Property Inquiry
        - Apartment Search
        - Tour Itinerary as a block (break into individual buildings)
        - Interaction summaries
        - Demographic or client profile data

        ✅ Output must be a single clean JSON list like:
        [
          {
            "building_name": "The Parker",
            "tour_time": "10:00 AM",
            "status": "toured",
            "actions": ["Guest Card", "Call"],
            "booking_method": "Call",
            "tour_type": "In-Person Tour",
            "notes": null,
            "timestamp": "2025-04-23T21:28:22.000+00:00"
          }
        ]
    """),
    expected_output=dedent("""
[
  {
    "building_name": "Example Tower",
    "tour_time": "11:30 AM",
    "status": "rejected",
    "actions": ["Guest Card", "Call"],
    "booking_method": "Call",
    "tour_type": "Virtual Tour",
    "notes": "Client disliked previous experience",
    "timestamp": "2025-04-22T18:48:15.000+00:00"
  }
]
    """),
    markdown=True,
)



# === FUNCTION TO ANALYZE MESSAGES ===
def analyze_client_messages(client_messages, requirements):
    prompt = f"""
REQUIREMENTS:
{requirements}

CLIENT MESSAGES:
{client_messages}

Please analyze the messages and return structured JSON only.
"""
    try:
        print("Analyzing messages...")
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"
    
    
    

# === FUNCTION TO SAVE RESULTS TO CSV ===
# def save_results_to_csv(client_id, analysis_result):
#     """Save the analysis results into a CSV file for each client."""
#     # Ensure the result is a list of dictionaries (JSON-like structure)
#     if isinstance(analysis_result, str):
#         # Check for error or empty result before parsing
#         if not analysis_result.strip() or analysis_result.strip().lower().startswith("error"):
#             print(f"Skipping client {client_id}: No valid analysis result. Message: {analysis_result}")
#             return
#         import json
#         try:
#             analysis_result = json.loads(analysis_result)
#         except Exception as e:
#             print(f"Skipping client {client_id}: Failed to parse analysis result as JSON. Error: {e}\nRaw result: {analysis_result}")
#             return
#     # Convert the analysis result to a DataFrame
#     result_data = pd.DataFrame(analysis_result)
#     # Define filename
#     filename = f"client_{client_id}_analysis.csv"
#     result_data.to_csv(filename, index=False)
#     print(f"Results saved to {filename}")
    



def save_results_to_csv(client_id, analysis_result):
    """Save the analysis results into a CSV file for each client, cleaning non-JSON output."""
    import json
    import re

    if isinstance(analysis_result, str):
        cleaned = analysis_result.strip()

        # Remove triple backticks and ```json from beginning/end
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        # Try to find the first [ or { and cut out anything before it (for robustness)
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        # If it's empty or an error, skip
        if not cleaned or cleaned.lower().startswith("error"):
            print(f"Skipping client {client_id}: No valid analysis result. Message: {analysis_result}")
            return
        try:
            analysis_result = json.loads(cleaned)
        except Exception as e:
            print(f"Skipping client {client_id}: Failed to parse analysis result as JSON. Error: {e}\nRaw result: {analysis_result}")
            return

    # Convert the analysis result to a DataFrame
    result_data = pd.DataFrame(analysis_result)
    filename = f"client_{client_id}_analysis.csv"
    result_data.to_csv(filename, index=False)
    print(f"Results saved to {filename}")






# # === USAGE EXAMPLE ===
# if __name__ == "__main__":
#     sample_messages = """
#     === Complete Chat History ===
#     # Paste the chat history here
    
#     # Paste your chat here
#     """

#     sample_requirements = "Extract only structured building interaction data. Ignore property searches, apartment types, or summaries."

#     result = analyze_client_messages(sample_messages, sample_requirements)

#     print("\nANALYSIS RESULTS:")
#     print(result)


# === MAIN PROCESS ===
def process_clients():
    # Step 1: Connect to the database
    conn = connect_to_db()

    # Step 2: Fetch clients whose current_stage is 4 or greater
    clients = fetch_qualified_clients(conn)

    # Step 3: Process only the first 10 clients
    for client in clients[:1]:
        client_id = client[0]  # Extracting client_id from the tuple
        print(f"Processing client {client_id}...")

        # Fetch the client's chat messages
        client_messages = fetch_client_messages(conn, client_id)

        # Combine messages into a single string (you can adjust this as needed)
        all_messages = "\n".join(client_messages)

            # *** LOG THE INPUT YOU'RE SENDING TO GEMINI ***
        print(f"\n\n=== CLIENT {client_id} MESSAGES ===\n{all_messages}\n")
        # Define requirements for analysis
        # requirements = "Extract structured building interaction data only. Ignore property searches, apartment types, or summaries."

        requirements = """
Extract ONLY building-level interactions, one JSON object per building per action.
– Each object must have:
    • building_name (exact text from the chat)
    • action (tour_scheduled | tour_cancelled | replaced | etc.)
    • timestamp (ISO or “none” if not mentioned)
    • notes (if any free-text reason)
Example output for these sample messages:

    “Client: Let’s schedule a tour of The Parker at 10AM tomorrow.”
    “Agent: Confirmed. See you at The Parker.”

should return:

[
  {
    "building_name": "The Parker",
    "action": "tour_scheduled",
    "timestamp": "2025-06-05T10:00:00-05:00",
    "notes": null
  }
]

Now, ANALYZE the following CHAT MESSAGES and return exactly a JSON array matching that schema—nothing else.
"""

        

        
        # Step 4: Analyze client messages using AI agent
        analysis_result = analyze_client_messages(all_messages, requirements)
        
        # in process_clients(), right after analysis_result = ...
        print(f"\nRAW GEMINI RESPONSE for client {client_id}:\n{analysis_result}\n")


        # Step 5: Save the analysis result in a CSV file
        save_results_to_csv(client_id, analysis_result)

    # Close the database connection
    conn.close()


# === RUN THE PROCESS ===
if __name__ == "__main__":
    process_clients()

