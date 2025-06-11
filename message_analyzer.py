import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
import psycopg2
from urllib.parse import urlparse
import pandas as pd
from textwrap import dedent
import google.generativeai as genai
from agno.agent import Agent
from agno.tools.exa import ExaTools
import dateutil.parser as dp

load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")

genai.configure(api_key=GENAI_API_KEY)

STATE_FILE = "last_run.json"


def load_last_run():
    try:
        with open(STATE_FILE, "r") as f:
            ts = json.load(f)["last_run"]
            return datetime.fromisoformat(ts)
    except Exception:
        # If missing or malformed, default far in the past
        return datetime(2000, 1, 1, tzinfo=timezone.utc)


def save_last_run(ts: datetime):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_run": ts.astimezone(timezone.utc).isoformat()}, f)


def normalize_timestamp(ts_str):
    """Normalize timestamp string to ISO format or 'none'."""
    if not ts_str or str(ts_str).lower() == "none":
        return "none"
    try:
        dt = dp.parse(ts_str)
    
    except Exception:
        return "none"
    
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


def fetch_qualified_clients(conn, since=None):
    """Fetch clients whose current_stage is ≥4 and created_on ≥ Jan 1, 2025, sorted oldest first."""
    query = """
      SELECT client_id, created_on
      FROM client_stage_progression
      WHERE current_stage = 4
    """
    params = []
    if since is not None:
        query += " AND created_on > %s"
        params.append(since)
    else:
        query += " AND created_on >= '2025-01-01'"
    query += " ORDER BY created_on ASC"
    with conn.cursor() as cur:
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)
        # Remove duplicates while preserving order
        seen = set()
        client_ids = []
        for row in cur.fetchall():
            client_id = row[0]
            if client_id not in seen:
                seen.add(client_id)
                client_ids.append(client_id)
        return client_ids


def process_incremental():
    # 1) connect
    conn = connect_to_db()
    if not conn:
        print("DB connection failed")
        return

    try:
        # 2) load watermark
        last_run = load_last_run()
        print(f"▶️  Last run: {last_run.isoformat()}")

        # 3) fetch only new clients
        new_clients = fetch_qualified_clients(conn, last_run)
        print(f"🔍  Found {len(new_clients)} new clients since last run")
        print(f"[BATCH DEBUG] Total clients to process: {len(new_clients)}")

        processed_count = 0
        # 4) process each
        for idx, client_id in enumerate(new_clients, 1):
            # Check if connection is closed, and reconnect if needed
            if conn.closed:
                print("[INFO] Database connection lost. Reconnecting...")
                conn = connect_to_db()
                if not conn:
                    print("[ERROR] Could not reconnect to the database. Exiting batch.")
                    break
            if os.path.exists("all_building_interactions.csv"):
                try:
                    df_existing = pd.read_csv("all_building_interactions.csv", usecols=["client_id"])
                    if str(client_id) in set(df_existing["client_id"].astype(str)):
                        print(f"[SKIP] Client {client_id} already processed. Skipping.")
                        continue
                    
                except Exception as e:
                    print(f"Warning: Could not check processed client IDs: {e}")
            print(f"— Processing client {client_id}")
            messages = fetch_client_messages(conn, client_id)
            all_msgs = "\n".join(messages)
            requirements = dedent(
                """
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
                - sales_agent
                - leasing_agent
                - building_address
                - unit_number
                - price
                - bed_bath
                - client_feedback
                - agent_notes
                - appointment_status
                - appointment_location
                - parking_info
                - pet_policy
                - application_status
                - follow_up_required
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
            """
            )
            raw = analyze_client_messages(all_msgs, requirements)
            save_results_to_csv(client_id, raw)
            processed_count += 1
            print(f"[BATCH DEBUG] Finished processing client {client_id} ({processed_count}/{len(new_clients)})")

        # ── Insert KPI summary here ──
        try:
            master_df = pd.read_csv("all_building_interactions.csv")
            total_clients     = master_df["client_id"].nunique()
            total_rows        = len(master_df)
            tours_scheduled   = (master_df["tour_status"] == "toured").sum()
            tours_completed   = (master_df["tour_completed"] == True).sum()
            unique_buildings  = master_df["building_name"].nunique()
            print(
                f"👉 Overall: {total_clients} clients, {total_rows} rows; "
                f"{tours_scheduled} scheduled, {tours_completed} completed, "
                f"{unique_buildings} unique buildings."
            )
        except FileNotFoundError:
            print("👉 No master CSV found yet—skipping KPI summary.")

        # 5) bump watermark
        now = datetime.now(timezone.utc)
        save_last_run(now)
        print(f"✅  Updated last_run to {now.isoformat()}")

    finally:
        conn.close()


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
                print(
                    f"Fetched {len(messages)} messages from 'public.textmessage' for client {client_id}."
                )
                return messages
            print(
                f"No messages found in 'public.textmessage' for client {client_id}, checking 'client_fub_messages'..."
            )
            query2 = """
                SELECT message
                FROM client_fub_messages
                WHERE client_id = %s
            """
            cur.execute(query2, (client_id,))
            messages2 = [row[0] for row in cur.fetchall() if row[0] is not None]
            if messages2:
                print(
                    f"Fetched {len(messages2)} messages from 'client_fub_messages' for client {client_id}."
                )
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
            prompt = last.content if hasattr(last, "content") else last.get("content", str(last))
        else:
            prompt = messages
        model = genai.GenerativeModel("gemini-2.0-flash")
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
    description=dedent(
        """
        You manage a team of agents who collect client requirements.
        Your primary goal is to track all building-related interactions:
        - Buildings sent to clients
        - Client responses: acceptance, rejection (with reason), or replacement requests
        - Full history of interaction with timestamps and actions (tour scheduled, canceled, etc.)
    """
    ),
    instructions=dedent(
        """
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
    """
    ),
    expected_output=dedent(
        """
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
    """
    ),
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
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


def save_results_to_csv(client_id, analysis_result, master_csv="all_building_interactions.csv"):
    """Save the analysis results into a per-client CSV and append to a master CSV, cleaning non-JSON output."""
    import json, re

    # Step 0: Load processed client IDs if master CSV exists
    processed_clients = set()
    if os.path.exists(master_csv):
        try:
            df_existing = pd.read_csv(master_csv, usecols=["client_id"])
            processed_clients = set(df_existing["client_id"].astype(str))
        except Exception as e:
            print(f"Warning: Could not load processed client IDs from master CSV: {e}")

    # Step 1: clean up any ```json wrappers, extract valid JSON
    if isinstance(analysis_result, str):
        cleaned = analysis_result.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        # Try to extract the first JSON array/object (robust, even with extra text)
        m = re.search(r"(\[.*?\]|\{.*?\})", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(0)
        if not cleaned or cleaned.lower().startswith("error"):
            print(f"No valid analysis result for client {client_id}. Writing error message to CSV.")
            os.makedirs("csv", exist_ok=True)
            client_filename = os.path.join("csv", f"{client_id}_analysis.csv")
            with open(client_filename, "w", encoding="utf-8") as f:
                f.write("error: No valid analysis result from Gemini. Raw output: " + repr(analysis_result))
            return None
        try:
            records = json.loads(cleaned)
        except Exception as e:
            print(f"JSON parse error for client {client_id}: {e}. Writing raw Gemini output to CSV.")
            os.makedirs("csv", exist_ok=True)
            client_filename = os.path.join("csv", f"{client_id}_analysis.csv")
            with open(client_filename, "w", encoding="utf-8") as f:
                f.write(f"error: JSON parse error: {e}\nRaw Gemini output: {repr(analysis_result)}")
            return None
    else:
        records = analysis_result

    # Step 2: ensure all keys & normalize, add new columns
    for rec in records:
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
        rec.setdefault("sales_agent", "none")
        rec.setdefault("leasing_agent", "none")
        rec.setdefault("building_address", "none")
        rec.setdefault("unit_number", "none")
        rec.setdefault("price", "none")
        rec.setdefault("bed_bath", "none")
        rec.setdefault("client_feedback", "none")
        rec.setdefault("agent_notes", "none")
        rec.setdefault("appointment_status", "none")
        rec.setdefault("appointment_location", "none")
        rec.setdefault("parking_info", "none")
        rec.setdefault("pet_policy", "none")
        rec.setdefault("application_status", "none")
        rec.setdefault("follow_up_required", False)
        # normalize any timestamps
        rec["sent_date"] = normalize_timestamp(rec["sent_date"])
        rec["tour_date"] = normalize_timestamp(rec["tour_date"])
        rec["timestamp"] = normalize_timestamp(rec.get("timestamp"))

    # Step 3: build DataFrame and tag client_id
    df = pd.DataFrame(records)
    df.insert(0, "client_id", client_id)

    # Step 4: save per-client file
    os.makedirs("csv", exist_ok=True)
    client_filename = os.path.join("csv", f"{client_id}_analysis.csv")
    df.to_csv(client_filename, index=False)
    print(f"Results saved to {client_filename}")

    # Step 5: append to master only if not already processed
    if str(client_id) in processed_clients:
        print(f"[SKIP] Client {client_id} already processed in master CSV. Skipping append.")
        return df
    write_header = not os.path.exists(master_csv)
    df.to_csv(master_csv, mode="a", header=write_header, index=False)
    if write_header:
        print(f"Created master file and appended data for client {client_id}.")
    else:
        print(f"Appended data for client {client_id} to master file.")
        
    return df


def enhanced_requirements():
    return dedent(
        '''
        Extract every building mentioned in the CLIENT MESSAGES, even if details are partial, ambiguous, or only referenced indirectly (e.g., by nickname, address, or context). For each building, output a JSON object with these keys:
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
        - sales_agent
        - leasing_agent
        - building_address
        - unit_number
        - price
        - bed_bath
        - client_feedback
        - agent_notes
        - appointment_status
        - appointment_location
        - parking_info
        - pet_policy
        - application_status
        - follow_up_required
        Rules:
        • Include a record for every building mentioned, even if some fields are "none".
        • Look for building names and info in ALL messages (agent and client), not just agent messages.
        • If a building is referenced by nickname, address, or partial name, include it and note ambiguity in notes.
        • If a tour, rejection, or other info is mentioned later, link it back to the correct building if possible (use context and inference).
        • For missing info, use "none" (for strings), false (for booleans), or 0 (for numbers).
        • If multiple buildings are sent together, create a record for each.
        • If a building is mentioned multiple times, merge info into a single record if possible (combine all clues).
        • Only output the JSON array—no extra text.
        • Set `tour_completed` to true if the chat confirms the tour occurred, false otherwise.
        • If it did occur, set `tour_format` to one of "In-Person", "Virtual (Google Meet)", or "Drive-By". If format isn’t mentioned, default to "none".
        • Add as much detail as possible to notes if info is ambiguous, inferred, or scattered.
        • For new fields, extract if present or inferable, otherwise use "none" or false.
        • If a field can be inferred from context (e.g., price, agent, feedback), fill it in.
        • If a building is referenced indirectly (e.g., "the first one you sent"), try to resolve it and link info.
        • If a message refers to multiple buildings, split and create a record for each.
        • If a building is mentioned in a group or as part of a list, create a record for each.
        Example:
        Chat:
          "Agent: Here’s The Parker, 123 Main St."
          "Agent: Let’s tour The Parker at 10AM tomorrow."
          "Client: I didn’t like the area, let’s skip it."
          "Agent: Also, The Summit and The Grove are available."
          "Client: Can we see The Grove on Friday?"
          "Agent: The one with the pool is $2,000/mo."
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
            "timestamp": "2025-05-01T14:30:00-05:00",
            "sales_agent": "none",
            "leasing_agent": "none",
            "building_address": "123 Main St.",
            "unit_number": "none",
            "price": "none",
            "bed_bath": "none",
            "client_feedback": "didn’t like the area",
            "agent_notes": "none",
            "appointment_status": "none",
            "appointment_location": "none",
            "parking_info": "none",
            "pet_policy": "none",
            "application_status": "none",
            "follow_up_required": false
          },
          {
            "building_name": "The Summit",
            "sent_date": "2025-05-01T14:31:00-05:00",
            "sent_method": "Call",
            "tour_status": "sent",
            "tour_date": "none",
            "tour_time": "none",
            "tour_type": "none",
            "rejection_reason": "none",
            "replacement_requested": false,
            "notes": "No further info",
            "timestamp": "2025-05-01T14:31:00-05:00",
            "sales_agent": "none",
            "leasing_agent": "none",
            "building_address": "none",
            "unit_number": "none",
            "price": "none",
            "bed_bath": "none",
            "client_feedback": "none",
            "agent_notes": "none",
            "appointment_status": "none",
            "appointment_location": "none",
            "parking_info": "none",
            "pet_policy": "none",
            "application_status": "none",
            "follow_up_required": false
          },
          {
            "building_name": "The Grove",
            "sent_date": "2025-05-01T14:31:00-05:00",
            "sent_method": "Call",
            "tour_status": "toured",
            "tour_date": "2025-05-02",
            "tour_time": "none",
            "tour_type": "none",
            "rejection_reason": "none",
            "replacement_requested": false,
            "notes": "Tour requested by client for Friday",
            "timestamp": "2025-05-01T14:31:00-05:00",
            "sales_agent": "none",
            "leasing_agent": "none",
            "building_address": "none",
            "unit_number": "none",
            "price": "$2,000/mo",
            "bed_bath": "none",
            "client_feedback": "none",
            "agent_notes": "none",
            "appointment_status": "none",
            "appointment_location": "none",
            "parking_info": "none",
            "pet_policy": "none",
            "application_status": "none",
            "follow_up_required": false
          }
        ]
        Now ANALYZE the following messages and return exactly that schema:
        '''
    )

def postprocess_building_records(records):
    """
    Post-process the extracted building records to merge duplicates, fill missing info, and improve coverage.
    Also print debug info about what was found.
    """
    from collections import defaultdict
    merged = defaultdict(dict)
    for rec in records:
        name = rec.get("building_name", "none").strip().lower()
        if name in merged:
            for k, v in rec.items():
                if k not in merged[name] or merged[name][k] in (None, "none", False, "Not mentioned"):
                    merged[name][k] = v
                elif v not in (None, "none", False, "Not mentioned"):
                    merged[name][k] = v
        else:
            merged[name] = rec.copy()
    # Debug printout
    print("[DEBUG] Buildings extracted:")
    for bname, rec in merged.items():
        print(f"  - {rec.get('building_name','none')}: {[k for k,v in rec.items() if v not in ('none', False, '', None, 'Not mentioned')]}")
    return list(merged.values())


def process_single_client(client_id):
    """Process a single client by client_id: fetch messages, analyze, and save results."""
    conn = connect_to_db()
    if not conn:
        print("DB connection failed")
        return
    try:
        messages = fetch_client_messages(conn, client_id)
        if not messages:
            print(f"No messages found for client {client_id}.")
            return
        all_msgs = "\n".join(messages)
        requirements = enhanced_requirements()
        raw = analyze_client_messages(all_msgs, requirements)
        # Try to parse and postprocess
        import json
        try:
            records = json.loads(raw)
            records = postprocess_building_records(records)
        except Exception:
            records = raw
        save_results_to_csv(client_id, records)
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "single":
        process_single_client(sys.argv[2])
    else:
        process_incremental()

