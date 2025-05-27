import os
from dotenv import load_dotenv

load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")

import psycopg2
from urllib.parse import urlparse
import pandas as pd
from datetime import datetime
from textwrap import dedent
import google.generativeai as genai
from agno.agent import Agent
from agno.tools.exa import ExaTools
import dateutil.parser as dp

genai.configure(api_key=GENAI_API_KEY)


def normalize_timestamp(ts_str):
    """Normalize timestamp string to ISO format or 'none'."""
    if not ts_str or str(ts_str).lower() == "none":
        return "none"
    dt = dp.parse(ts_str)
    if dt.time() == datetime.min.time():
        return "none"
    return dt.isoformat()


today = datetime.now().strftime("%Y-%m-%d")


def connect_to_db():
    """Connect to the PostgreSQL database using the provided connection string."""
    db_url = (
        "postgres://readonly:p2e0dfd8702aac2f2b98b80f2e14430fb7d3cec6f8aec1770701283042b112712@ec2-23-20-93-193.compute-1.amazonaws.com:5432/d5pt3225ki095v"
    )
    result = urlparse(db_url)
    try:
        conn = psycopg2.connect(
            dbname=result.path[1:],
            user=result.username,
            password=result.password,
            host=result.hostname,
            port=result.port,
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None


def fetch_qualified_clients(conn):
    """Fetch clients whose current_stage is 4 or greater."""
    query = """
        SELECT client_id 
        FROM client_stage_progression 
        WHERE current_stage >= 4
    """
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


def fetch_client_messages(conn, client_id):
    """
    Fetch messages for a specific client:
    - Try from 'public.textmessage' table (status in 'Sent', 'Received').
    - If not found, fall back to 'client_fub_messages' table.
    Returns a list of message strings (in order).
    """
    with conn.cursor() as cur:
        try:
            query1 = """
                SELECT message
                FROM public.textmessage
                WHERE client_id = %s
                  AND status IN ('Sent', 'Received')
                ORDER BY created ASC
            """
            cur.execute(query1, (client_id,))
            rows = cur.fetchall()
            messages = [row[0] for row in rows if row[0] is not None]
            if messages:
                print(f"Fetched {len(messages)} messages from 'public.textmessage' for client {client_id}.")
                return messages
            print(f"No messages found in 'public.textmessage' for client {client_id}, checking 'client_fub_messages'...")
            query2 = """
                SELECT message
                FROM client_fub_messages
                WHERE client_id = %s
            """
            cur.execute(query2, (client_id,))
            messages2 = [row[0] for row in cur.fetchall() if row[0] is not None]
            if messages2:
                print(f"Fetched {len(messages2)} messages from 'client_fub_messages' for client {client_id}.")
            else:
                print(f"No messages found for client {client_id} in either table.")
            return messages2
        except Exception as e:
            print(f"Error fetching client messages for client {client_id}: {e}")
            return []


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


def analyze_client_messages(client_messages, requirements):
    """Analyze client messages using Gemini AI and return structured JSON."""
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


def save_results_to_csv(client_id, analysis_result):
    """Save the analysis results into a CSV file for each client, cleaning non-JSON output."""
    import json
    import re
    if isinstance(analysis_result, str):
        cleaned = analysis_result.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        if not cleaned or cleaned.lower().startswith("error"):
            print(f"Skipping client {client_id}: No valid analysis result. Message: {analysis_result}")
            return
        try:
            analysis_result = json.loads(cleaned)
            for rec in analysis_result:
                # Ensure all keys exist
                rec.setdefault("building_name", "none")
                rec.setdefault("sent_date", "none")
                rec.setdefault("sent_method", "Not mentioned")
                rec.setdefault("tour_status", "not toured")
                rec.setdefault("tour_completed", False)
                rec.setdefault("tour_format", "none")
                rec.setdefault("tour_date", "none")
                rec.setdefault("tour_time", "none")
                rec.setdefault("tour_type", "none")
                rec.setdefault("rejection_reason", "none")
                rec.setdefault("replacement_requested", False)
                rec.setdefault("notes", "none")
                # Normalize timestamps
                rec["sent_date"] = normalize_timestamp(rec["sent_date"])
                rec["tour_date"] = normalize_timestamp(rec["tour_date"])
                rec["timestamp"] = normalize_timestamp(rec.get("timestamp"))
                # tour_time and others remain as-is (string/"none")
        except Exception as e:
            print(f"Skipping client {client_id}: Failed to parse analysis result as JSON. Error: {e}\nRaw result: {analysis_result}")
            return
    result_data = pd.DataFrame(analysis_result)
    filename = f"client_{client_id}_analysis.csv"
    result_data.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def process_clients():
    """Main process to fetch, analyze, and save client message data."""
    conn = connect_to_db()
    if conn is None:
        print("Database connection failed. Exiting.")
        return
    try:
        clients = fetch_qualified_clients(conn)
        for client in clients[:5]:  # Only process the first client for now
            client_id = client[0]
            print(f"Processing client {client_id}...")
            client_messages = fetch_client_messages(conn, client_id)
            all_messages = "\n".join(client_messages)
            requirements = dedent("""
                Extract detailed building interactions from these CLIENT MESSAGES.
                Output a JSON array where each element is a building object with exactly these keys:

                - building_name
                - sent_date
                - sent_method
                - tour_status
                - tour_completed
                - tour_format
                - tour_date
                - tour_time
                - tour_type
                - rejection_reason
                - replacement_requested
                - notes
                - timestamp

                Rules:
                • If the agent “sent” a building (gave its name/details), set sent_date to the chat timestamp and infer sent_method.
                • If there’s no explicit date, default sent_date▶︎“none”.
                • Fill tour_status based on whether the building was toured, canceled, rejected, or replaced.
                • For any missing info, use “none” (for strings) or false (for replacement_requested).
                • Only output this JSON array–no extra text.
                • Set `tour_completed` to true if the chat confirms the tour occurred, false otherwise.  
                • If it did occur, set `tour_format` to one of "In-Person", "Virtual (Google Meet)", or "Drive-By".  
                • If format isn’t mentioned, default to "none". 

                Example:
                Chat:
                  “Agent: Here’s The Parker, 123 Main St.”
                  “Agent: Let’s tour The Parker at 10AM tomorrow.”
                  “Client: I didn’t like the area, let’s skip it.”

                Output:
                [
                  {
                    "building_name": "The Parker",
                    "sent_date": "2025-05-01T14:30:00-05:00",
                    "sent_method": "Call",
                    "tour_status": "rejected",
                    "tour_date": "2025-05-02",
                    "tour_time": "10:00 AM",
                    "tour_type": "In-Person Tour",
                    "rejection_reason": "didn’t like the area",
                    "replacement_requested": false,
                    "notes": "Client skipped after tour",
                    "timestamp": "2025-05-01T14:30:00-05:00"
                  }
                ]

                Now ANALYZE the following messages and return exactly that schema:
            """)
            analysis_result = analyze_client_messages(all_messages, requirements)
            print(f"\nRAW GEMINI RESPONSE for client {client_id}:\n{analysis_result}\n")
            save_results_to_csv(client_id, analysis_result)
    finally:
        if conn is not None:
            conn.close()
        

if __name__ == "__main__":
    process_clients()

