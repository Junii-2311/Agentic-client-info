import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
import csv

SERVICE_ACCOUNT_FILE = 'credentials.json'
USER_EMAIL = 'ahmed.junaid@homeeasy.com'
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def main():
    # Load service account credentials
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    # Impersonate the user
    delegated_credentials = credentials.with_subject(USER_EMAIL)

    # Build the Gmail API client
    service = build('gmail', 'v1', credentials=delegated_credentials)

    # Use Gmail search query to find all emails sent or received by the USER_EMAIL
    query = f'from:{USER_EMAIL} OR to:{USER_EMAIL}'
    results = service.users().messages().list(userId='me', q=query, maxResults=10).execute()
    messages = results.get('messages', [])

    # Prepare to write to CSV
    with open('filtered_emails.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['From', 'To', 'Subject', 'Body']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print(f"All emails sent or received by {USER_EMAIL}:")
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
            # Decode from base64url
            import base64
            if body:
                body = base64.urlsafe_b64decode(body.encode('ASCII')).decode('utf-8', errors='replace')
                # Remove HTML tags if present
                import re
                body = re.sub('<[^<]+?>', '', body)
                body = body.strip()
            else:
                body = '(No Body Found)'
            print(f"From: {from_} | To: {to_} | Subject: {subject}\nBody: {body}\n{'-'*40}")
            writer.writerow({'From': from_, 'To': to_, 'Subject': subject, 'Body': body})

if __name__ == '__main__':
    main()
