import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
import base64
import re

SERVICE_ACCOUNT_FILE = 'service_file/credentials.json'
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def fetch_client_emails(agent_email, client_email, max_results=20):
    """
    Fetch emails between the agent and the client using Gmail API.
    Returns a list of formatted email strings (labelled as EMAIL).
    """
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    delegated_credentials = credentials.with_subject(agent_email)
    service = build('gmail', 'v1', credentials=delegated_credentials)
    # Search for emails between agent and client
    query = f'(from:{agent_email} to:{client_email}) OR (from:{client_email} to:{agent_email})'
    results = service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
    messages = results.get('messages', [])
    emails = []
    for msg in messages:
        msg_detail = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        headers = msg_detail.get('payload', {}).get('headers', [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '(No Subject)')
        from_ = next((h['value'] for h in headers if h['name'] == 'From'), '(No From)')
        to_ = next((h['value'] for h in headers if h['name'] == 'To'), '(No To)')
        # Extract the body (handle plain text and HTML)
        body = ''
        payload = msg_detail.get('payload', {})
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain':
                    body = part.get('body', {}).get('data', '')
                    break
                elif part.get('mimeType') == 'text/html' and not body:
                    body = part.get('body', {}).get('data', '')
        else:
            body = payload.get('body', {}).get('data', '')
        if body:
            body = base64.urlsafe_b64decode(body.encode('ASCII')).decode('utf-8', errors='replace')
            body = re.sub('<[^<]+?>', '', body)
            body = body.strip()
        else:
            body = '(No Body Found)'
        # Label and format
        email_str = f"EMAIL | From: {from_} | To: {to_} | Subject: {subject}\n{body}"
        emails.append(email_str)
    return emails

# No main() block needed for integration
