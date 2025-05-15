import os
import pickle
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time # Import time for potential retries

# --- Configuration ---
# Ensure BASE_DIR is consistent with your main Flask app
# If this file is in the same directory as the Flask app, BASE_DIR might need adjustment
# or be passed in, or determined dynamically.
# For simplicity, assuming it's run from a context where this path is correct.
# A better approach might be to define paths relative to this file's location.
# script_dir = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = os.path.join(script_dir, "..") # Example if gmail_api.py is in a subdir
BASE_DIR = "D:/cleanInboxAI" # Make sure this matches the Flask app's BASE_DIR
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json") # Path to your OAuth client secrets file
TOKEN_PATH = os.path.join(BASE_DIR, "token.pickle") # Path to store/load the user's access/refresh token
# Define the scopes needed. Adjust if necessary.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', # Read emails
          'https://www.googleapis.com/auth/gmail.modify',  # Delete, mark as read/unread, add/remove labels
          'https://www.googleapis.com/auth/gmail.labels'] # Manage labels (optional, but can be useful)

# --- Gmail Service Authentication ---
def get_gmail_service():
    """
    Authenticates the user via OAuth 2.0 and returns the Gmail API service object.
    Handles token loading, refreshing, and saving. Initiates OAuth flow if needed.
    """
    creds = None
    # Load existing token if available
    if os.path.exists(TOKEN_PATH):
        try:
            with open(TOKEN_PATH, 'rb') as token:
                creds = pickle.load(token)
            print(f"Token loaded successfully from {TOKEN_PATH}")
        except (EOFError, pickle.UnpicklingError, FileNotFoundError, Exception) as e:
             # Handle cases where the token file is corrupted or unreadable
             print(f"Error loading token file from {TOKEN_PATH}: {e}. Re-authentication will be required.")
             creds = None # Ensure creds is None so flow runs

    # If no valid credentials, try to refresh or run the OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # If token expired and refresh token exists, try refreshing
            try:
                print("Refreshing expired access token...")
                creds.refresh(Request())
                print("Token refreshed successfully.")
                # Save the refreshed credentials immediately
                try:
                    with open(TOKEN_PATH, 'wb') as token:
                        pickle.dump(creds, token)
                    print(f"Refreshed token saved to {TOKEN_PATH}")
                except IOError as e:
                    print(f"Error saving refreshed token to {TOKEN_PATH}: {e}")
            except Exception as e:
                # If refresh fails, delete the token file and force re-authentication
                print(f"Error refreshing token: {e}. Deleting potentially invalid token and re-authenticating.")
                if os.path.exists(TOKEN_PATH):
                    try:
                        os.remove(TOKEN_PATH)
                        print(f"Removed invalid token file: {TOKEN_PATH}")
                    except OSError as remove_err:
                        print(f"Error removing token file {TOKEN_PATH}: {remove_err}")
                creds = None # Force re-authentication flow
        # Only run the flow if creds are still None (initial run or failed refresh)
        if not creds:
            # Check if the credentials file exists before starting the flow
            if not os.path.exists(CREDENTIALS_PATH):
                 msg = f"CRITICAL ERROR: OAuth Credentials file not found at {CREDENTIALS_PATH}. Cannot authenticate."
                 print(msg)
                 # Raising an exception might be better than returning None depending on calling context
                 # raise FileNotFoundError(msg)
                 return None # Cannot proceed

            print("No valid token found or refresh failed, initiating OAuth flow...")
            # Use InstalledAppFlow for scripts/local apps.
            # For web applications (like Flask running on a server), you'd typically use WebServerFlow.
            # However, InstalledAppFlow works for local development/testing of the Flask app.
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
                # run_local_server opens a browser tab for user consent.
                # port=0 lets it pick a random available port.
                # Consider adding prompt='consent' to force consent screen and ensure refresh token is granted,
                # especially the first time or if scopes change.
                # creds = flow.run_local_server(port=0, prompt='consent')
                creds = flow.run_local_server(port=0)
                print("OAuth flow completed successfully.")
            except Exception as flow_err:
                 print(f"Error during OAuth authorization flow: {flow_err}")
                 return None # Cannot proceed without credentials

            # Save the newly obtained credentials for the next run
            try:
                with open(TOKEN_PATH, 'wb') as token:
                    pickle.dump(creds, token)
                print(f"New token saved to {TOKEN_PATH}")
            except IOError as e:
                print(f"Error saving new token to {TOKEN_PATH}: {e}")
                # Continue even if saving fails, but auth will be needed next time

    # Build and return the Gmail API service object using the valid credentials
    try:
        service = build('gmail', 'v1', credentials=creds)
        print("Gmail API service object created successfully.")
        return service
    except HttpError as error:
        # Handle errors during service build (e.g., API not enabled)
        print(f'An HTTP error occurred building the Gmail service: {error}')
        return None
    except Exception as e:
        # Catch any other unexpected errors during service build
        print(f'An unexpected error occurred building the Gmail service: {e}')
        return None

# --- Email Operations ---

def list_messages(service, query="", max_results=100):
    """
    Lists message IDs matching the specified query, handling pagination.

    Args:
        service: Authorized Gmail API service instance.
        query (str): Gmail search query string (e.g., 'is:unread in:inbox').
        max_results (int): Maximum total number of messages to return.

    Returns:
        list: A list of message dictionaries, each containing 'id' and 'threadId'.
              Returns an empty list if no messages are found or an error occurs.
    """
    all_messages = []
    page_token = None
    retrieved_count = 0
    print(f"Listing messages with query: '{query}', max_results={max_results}")

    try:
        while retrieved_count < max_results:
            # Determine how many results to request in this page
            request_size = min(100, max_results - retrieved_count) # Max 100 per page allowed by API
            if request_size <= 0: break # Should not happen if loop condition is correct, but good safeguard

            print(f"Requesting page with max {request_size} results...")
            response = service.users().messages().list(
                userId='me',
                q=query,
                maxResults=request_size,
                pageToken=page_token
            ).execute()

            messages_on_page = response.get('messages', [])
            if not messages_on_page:
                print("No more messages found on this page.")
                break # No messages found on this page

            all_messages.extend(messages_on_page)
            retrieved_count = len(all_messages)
            print(f"Retrieved {len(messages_on_page)} messages this page. Total so far: {retrieved_count}")

            # Check for next page token
            page_token = response.get('nextPageToken')
            if not page_token:
                print("No next page token found.")
                break # No more pages available

        print(f"Finished listing messages. Found {len(all_messages)} messages matching query '{query}'.")
        # Ensure we don't return more than max_results (should be handled by loop, but double-check)
        return all_messages[:max_results]

    except HttpError as error:
        print(f'An API error occurred during message listing: {error}')
        # Consider more specific error handling based on status code if needed
        return [] # Return empty list on error
    except Exception as e:
        print(f'An unexpected error occurred during message listing: {e}')
        return []


def get_message_details(service, msg_id):
    """
    Gets details for a single message ID (Subject, Snippet, Labels).

    Args:
        service: Authorized Gmail API service instance.
        msg_id (str): The ID of the message to retrieve.

    Returns:
        dict: A dictionary containing message details ('id', 'threadId', 'subject',
              'snippet', 'labelIds') or None if an error occurs.
    """
    try:
        # Using format='metadata' is efficient as we only need headers and snippet.
        # metadataHeaders specifies which headers to include (reduces payload size).
        msg = service.users().messages().get(
            userId='me',
            id=msg_id,
            format='metadata',
            metadataHeaders=['Subject'] # Explicitly request only the Subject header
        ).execute()

        # Parse the response to extract relevant information
        email_info = {
            'id': msg['id'],
            'threadId': msg.get('threadId'), # Thread ID can be useful
            'subject': '(No Subject)', # Default subject
            'snippet': msg.get('snippet', ''), # Snippet provided by Gmail
            'labelIds': msg.get('labelIds', []) # List of label IDs applied to the message
        }

        # Extract the Subject header value
        headers = msg.get('payload', {}).get('headers', [])
        for header in headers:
            # Case-insensitive comparison for header name
            if header.get('name', '').lower() == 'subject':
                email_info['subject'] = header.get('value', '(No Subject)')
                break # Found the subject, no need to check further headers

        # print(f"Successfully retrieved details for message {msg_id}")
        return email_info

    except HttpError as error:
        print(f'An API error occurred getting details for message {msg_id}: {error}')
        return None # Indicate failure
    except Exception as e:
        print(f'An unexpected error occurred getting details for message {msg_id}: {e}')
        return None


def get_batch_message_details(service, message_ids, batch_size=50):
    """
    Fetches details for multiple message IDs using efficient batch requests.

    Args:
        service: Authorized Gmail API service instance.
        message_ids (list): A list of message IDs (strings) to fetch details for.
        batch_size (int): The number of individual requests to include in each batch execution.
                          Max recommended by Google is 100, default is 50.

    Returns:
        list: A list of email detail dictionaries (same format as get_message_details).
              May contain fewer items than message_ids if errors occurred for some messages.
    """
    all_details = []
    if not message_ids:
        print("No message IDs provided for batch fetching.")
        return all_details

    print(f"Preparing to fetch details for {len(message_ids)} messages in batches of {batch_size}...")

    # Define the callback function to process results from each individual request in the batch
    def batch_callback(request_id, response, exception):
        nonlocal all_details # Allow modification of the outer scope list
        if exception:
            # Handle errors for individual requests within the batch
            print(f"ERROR in batch request ID {request_id}: {exception}")
            # You could log the failed message ID here if needed, e.g., by correlating request_id
        elif response:
            # Process successful response
            try:
                email_info = {
                    'id': response['id'],
                    'threadId': response.get('threadId'),
                    'subject': '(No Subject)',
                    'snippet': response.get('snippet', ''),
                    'labelIds': response.get('labelIds', [])
                }
                headers = response.get('payload', {}).get('headers', [])
                for header in headers:
                    if header.get('name', '').lower() == 'subject':
                        email_info['subject'] = header.get('value', '(No Subject)')
                        break
                all_details.append(email_info)
            except KeyError as e:
                print(f"KeyError processing batch response for request ID {request_id}: Missing key {e}. Response: {response}")
            except Exception as e:
                print(f"Unexpected error processing batch response for request ID {request_id}: {e}")

    # Create the initial batch request object
    batch = service.new_batch_http_request(callback=batch_callback)

    # Add individual message get requests to the batch
    for i, msg_id in enumerate(message_ids):
        request = service.users().messages().get(userId='me', id=msg_id, format='metadata', metadataHeaders=['Subject'])
        # Add the request to the current batch, using msg_id as request_id for easier error tracking
        batch.add(request, request_id=msg_id)

        # Execute the batch if it reaches the desired size or if it's the last message
        # The condition `(i + 1) % batch_size == 0` checks if the batch is full
        # The condition `(i + 1) == len(message_ids)` checks if this is the last message
        if (i + 1) % batch_size == 0 or (i + 1) == len(message_ids):
            current_batch_size = len(batch._requests) # Get the actual number of requests in this batch
            print(f"Executing batch request for {current_batch_size} messages (Total processed: {i+1})...")
            try:
                batch.execute()
                print(f"Batch executed successfully.")
                # Add a small delay after execution to help avoid rate limits, especially in loops
                time.sleep(0.2) # Adjust delay as needed
            except HttpError as error:
                # This catches errors during the execution of the entire batch (e.g., network issues, auth errors)
                print(f"FATAL ERROR executing batch request: {error}")
                # Depending on the error, you might want to stop or retry
            except Exception as e:
                 print(f"Unexpected error during batch execution: {e}")

            # Prepare a new batch object for the next set of requests *only if* there are more messages
            if (i + 1) < len(message_ids):
                 batch = service.new_batch_http_request(callback=batch_callback)


    print(f"Finished batch processing. Successfully fetched details for {len(all_details)} out of {len(message_ids)} requested messages.")
    return all_details


def modify_message_labels(service, msg_id, labels_to_add=None, labels_to_remove=None):
    """
    Adds or removes labels from a specific message.

    Args:
        service: Authorized Gmail API service instance.
        msg_id (str): The ID of the message to modify.
        labels_to_add (list, optional): A list of label IDs to add. Defaults to None.
        labels_to_remove (list, optional): A list of label IDs to remove. Defaults to None.

    Returns:
        bool: True if the modification was successful, False otherwise.
    """
    if not labels_to_add and not labels_to_remove:
        print(f"No labels specified to add or remove for message {msg_id}.")
        return False # Nothing to do

    # Construct the request body
    body = {}
    if labels_to_add:
        # Ensure it's a list
        body['addLabelIds'] = list(labels_to_add) if isinstance(labels_to_add, (list, tuple)) else [labels_to_add]
    if labels_to_remove:
        # Ensure it's a list
        body['removeLabelIds'] = list(labels_to_remove) if isinstance(labels_to_remove, (list, tuple)) else [labels_to_remove]

    try:
        service.users().messages().modify(userId='me', id=msg_id, body=body).execute()
        add_str = f"Added: {body.get('addLabelIds', 'None')}"
        rem_str = f"Removed: {body.get('removeLabelIds', 'None')}"
        print(f"Successfully modified labels for message {msg_id}. {add_str}, {rem_str}")
        return True
    except HttpError as error:
        print(f'An API error occurred modifying labels for message {msg_id}: {error}')
        return False
    except Exception as e:
        print(f'An unexpected error occurred modifying labels for message {msg_id}: {e}')
        return False


def mark_as_read(service, msg_id):
    """Marks a specific message as read by removing the 'UNREAD' label."""
    print(f"Attempting to mark message {msg_id} as read...")
    return modify_message_labels(service, msg_id, labels_to_remove=['UNREAD'])

def mark_as_unread(service, msg_id):
     """Marks a specific message as unread by adding the 'UNREAD' label."""
     print(f"Attempting to mark message {msg_id} as unread...")
     return modify_message_labels(service, msg_id, labels_to_add=['UNREAD'])


def delete_message(service, msg_id):
    """
    Moves the specified message to the Trash folder.
    Note: This is generally preferred over permanent deletion via delete().
    """
    print(f"Attempting to move message {msg_id} to Trash...")
    try:
        # Use the trash method to move to trash
        service.users().messages().trash(userId='me', id=msg_id).execute()
        print(f"Successfully moved message {msg_id} to Trash.")
        return True
    except HttpError as error:
        print(f'An API error occurred moving message {msg_id} to trash: {error}')
        return False
    except Exception as e:
        print(f'An unexpected error occurred moving message {msg_id} to trash: {e}')
        return False

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    print("--- Testing Gmail API Module ---")
    print("Attempting to get Gmail service...")
    service = get_gmail_service()

    if service:
        print("\n--- Gmail Service Obtained ---")

        # --- Test 1: List Messages ---
        print("\n[Test 1] Listing UNREAD messages in INBOX (max 5)...")
        # Use a specific query, e.g., unread in primary inbox
        unread_message_ids_list = list_messages(service, query='is:unread in:inbox category:primary', max_results=5)

        if unread_message_ids_list:
            print(f"Found {len(unread_message_ids_list)} unread messages.")
            ids_only = [msg['id'] for msg in unread_message_ids_list]

            # --- Test 2: Get Batch Details ---
            print(f"\n[Test 2] Fetching details for {len(ids_only)} messages using batch...")
            unread_emails = get_batch_message_details(service, ids_only, batch_size=10) # Use smaller batch for testing

            print("\n--- Unread Email Details ---")
            for i, email in enumerate(unread_emails):
                print(f"  {i+1}. ID: {email.get('id')}, Subject: '{email.get('subject', '')[:60]}...' Labels: {email.get('labelIds')}")

                # --- Test 3 & 4: Modify Labels (Example - Uncomment carefully!) ---
                # IMPORTANT: These actions modify your actual Gmail inbox.
                # if i == 0: # Example: Mark the first email as read
                #     print(f"  [Test 3 - Action] Attempting to mark email {email['id']} as read...")
                #     success = mark_as_read(service, email['id'])
                #     print(f"  Mark as read result: {'Success' if success else 'Failed'}")
                #     # Example: Mark it back as unread
                #     # print(f"  Attempting to mark email {email['id']} back as unread...")
                #     # success_unread = mark_as_unread(service, email['id'])
                #     # print(f"  Mark as unread result: {'Success' if success_unread else 'Failed'}")


                # --- Test 5: Delete Message (Example - Uncomment very carefully!) ---
                # IMPORTANT: This moves the email to your Trash folder.
                # if i == 1: # Example: Delete the second email
                #      print(f"  [Test 5 - Action] Attempting to delete email {email['id']} (move to trash)...")
                #      success = delete_message(service, email['id'])
                #      print(f"  Delete result: {'Success' if success else 'Failed'}")

        else:
            print("No unread messages found matching the query.")

        # --- Test 6: Get Single Message Detail (Example) ---
        # Use an ID known to exist, perhaps from the list above if it wasn't deleted
        # if unread_message_ids_list:
        #     test_id = unread_message_ids_list[0]['id']
        #     print(f"\n[Test 6] Getting details for single message ID: {test_id}")
        #     single_detail = get_message_details(service, test_id)
        #     if single_detail:
        #         print(f"  Subject: {single_detail.get('subject')}")
        #         print(f"  Snippet: {single_detail.get('snippet')}")
        #         print(f"  Labels: {single_detail.get('labelIds')}")
        #     else:
        #         print(f"  Failed to get details for message {test_id}")

    else:
        print("\n--- Failed to Obtain Gmail Service ---")
        print("Please check:")
        print(f"1. If '{CREDENTIALS_PATH}' exists and is valid.")
        print(f"2. If '{TOKEN_PATH}' needs to be deleted for re-authentication.")
        print("3. Your internet connection and Google API Console settings.")

    print("\n--- Gmail API Module Test Finished ---")

