import os
import pickle
import json
import numpy as np
import re # Thêm thư viện regular expression
import string # Thêm thư viện xử lý chuỗi (cho dấu câu)
import traceback # For detailed error logging
import time # Added for logging timestamp
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for servers
import matplotlib.pyplot as plt

from flask import Flask, render_template, jsonify, send_file, request, url_for, redirect
from sklearn.model_selection import train_test_split
# --- Import different models ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC # More efficient than SVC for text data
# ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
)

# --- Import Google API Client Error ---
try:
    from googleapiclient.errors import HttpError
except ImportError:
    print("WARNING: googleapiclient not installed. Install it (`pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib`) to use Gmail features.")
    # Define a dummy HttpError if the library is not installed
    class HttpError(Exception):
        def __init__(self, resp, content, uri=None):
            self.resp = resp
            self.content = content
            self.uri = uri
            super().__init__(f"HTTP error {resp.status if hasattr(resp, 'status') else 'unknown'}")


# --- Import Gmail API Functions ---
# Attempt to import the custom Gmail API module
try:
    from gmail_api import (
        get_gmail_service,
        list_messages,
        get_batch_message_details,
        delete_message,
        mark_as_read,
        modify_message_labels # Now needed for untrash
    )
    GMAIL_API_AVAILABLE = True
    print("Successfully imported gmail_api functions.")
except ImportError as e:
    # Handle the case where the module is not found
    print(f"WARNING: Failed to import from gmail_api.py: {e}. Email processing features will be disabled.")
    GMAIL_API_AVAILABLE = False
    # Define dummy functions to prevent crashes if the API is unavailable
    def get_gmail_service(): return None
    def list_messages(service, query="", max_results=100): return []
    def get_batch_message_details(service, message_ids, batch_size=50): return []
    def delete_message(service, msg_id): print(f"DUMMY: Delete {msg_id}"); return False
    def mark_as_read(service, msg_id): print(f"DUMMY: Mark as read {msg_id}"); return False
    def modify_message_labels(service, msg_id, labels_to_add=None, labels_to_remove=None): print(f"DUMMY: Modify labels {msg_id}"); return False

# --- Configuration ---
app = Flask(__name__)

# Consider using relative paths or environment variables for better portability
# BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Example relative path
BASE_DIR = "D:/cleanInboxAI" # Base directory for storing files
MODEL_PATH = os.path.join(BASE_DIR, "email_predictor.pkl") # Path for the tuple (vectorizer, model)
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json") # Path for model evaluation metrics
PLOT_PATH = os.path.join(BASE_DIR, "roc_curve.png") # Path for the ROC curve plot
RL_FEEDBACK_LOG = os.path.join(BASE_DIR, "rl_feedback.log") # Simple log file for reinforcement learning feedback

# --- Constants for Email Processing ---
PROCESS_EMAIL_LIMIT = 50 # Max number of unread emails to process in one go
CONFIDENCE_THRESHOLD = 0.90 # Threshold for auto-keep/delete (e.g., 90% confidence)

# Ensure the base directory exists, create it if not
try:
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Ensured base directory exists: {BASE_DIR}")
except OSError as e:
    print(f"ERROR: Could not create base directory {BASE_DIR}: {e}")
    # Depending on the severity, you might want to exit or handle this differently
    GMAIL_API_AVAILABLE = False # Disable features requiring file storage


# --- Helper Functions ---

def load_model_and_vectorizer(path=MODEL_PATH):
    """Loads the saved vectorizer and model tuple."""
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                # Ensure both vectorizer and model are loaded
                loaded_data = pickle.load(f)
                if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                    vectorizer, model = loaded_data
                    print(f"Vectorizer and model loaded successfully from {path}")
                    return vectorizer, model
                else:
                    print(f"Error: Expected a tuple of (vectorizer, model) in {path}, but got {type(loaded_data)}")
                    return None, None
        except (pickle.UnpicklingError, EOFError, FileNotFoundError, TypeError, Exception) as e:
            print(f"Error loading model/vectorizer from {path}: {e}")
            traceback.print_exc() # Print full traceback for debugging
            return None, None
    else:
        print(f"Model file not found at {path}")
        return None, None

def log_rl_feedback(email_id, subject, features, predicted_action, confirmed_action):
     """Logs data point (user feedback)."""
     # (Implementation unchanged)
     try:
         with open(RL_FEEDBACK_LOG, "a") as f:
             log_entry = json.dumps({
                 "timestamp": time.time(),
                 "email_id": email_id,
                 "subject": subject if subject else '(No Subject Provided)',
                 "predicted_action": predicted_action, # Can be None
                 "confirmed_action": confirmed_action # 'keep' or 'delete'
             })
             f.write(log_entry + "\n")
     except IOError as e:
         print(f"Error writing to RL feedback log ({RL_FEEDBACK_LOG}): {e}")
     except Exception as e:
         print(f"An unexpected error occurred while writing to RL feedback log: {e}")


# --- Hàm tiền xử lý văn bản chi tiết theo yêu cầu ---
# (Copied from train_model_detailed_preprocessing_vi artifact)
def preprocess_text_detailed(text):
    """
    Áp dụng các bước tiền xử lý chi tiết cho văn bản email theo thứ tự yêu cầu.
    """
    if not isinstance(text, str):
        return "" # Trả về chuỗi rỗng nếu đầu vào không phải là string

    # Bước 1: Lowercasing
    text = text.lower()

    # Bước 2: Chuyển tiếng Anh thành tiếng Việt (Diễn giải: Chuẩn bị xử lý TV)
    # Không có hành động cụ thể ở đây, các bước sau áp dụng chung.

    # Bước 5: Loại bỏ ký tự đặc biệt và thẻ HTML
    text = re.sub('<[^>]*>', ' ', text) # Loại bỏ thẻ HTML trước

    # Bước 6: Xử lý URL và địa chỉ Email
    text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '<EMAIL>', text)

    # Bước 7: Xử lý số (Handling Numbers)
    text = re.sub(r'\d+', '<NUMBER>', text)

    # Bước 4: Loại bỏ dấu câu (Punctuation Removal)
    punctuation_to_remove = string.punctuation.replace('<', '').replace('>', '')
    text = text.translate(str.maketrans('', '', punctuation_to_remove))

    # Bước 5 (tiếp): Loại bỏ ký tự đặc biệt còn sót lại (nếu cần)
    # text = re.sub(r'[^\w\s<>]', '', text) # Cẩn thận với ký tự tiếng Việt

    # Bước 3: Tokenization (Thường do Vectorizer đảm nhiệm)
    # CountVectorizer/TfidfVectorizer sẽ thực hiện tokenization.

    # Cuối cùng: Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Flask Routes ---

@app.route("/")
def index():
    """Serves the main index page."""
    print("Serving index page.")
    # Check if templates folder exists and index.html is inside
    if not os.path.exists(os.path.join(app.template_folder, 'index.html')):
        return "Error: index.html template not found.", 404
    return render_template("index.html")

@app.route("/model_management")
def model_management():
    """Serves the model management page."""
    print("Serving model management page.")
    if not os.path.exists(os.path.join(app.template_folder, 'model_management.html')):
        return "Error: model_management.html template not found.", 404
    return render_template("model_management.html")

@app.route("/predict_email")
def predict_email_page():
    """Serves the email prediction/processing page."""
    print("Serving email prediction page.")
    if not os.path.exists(os.path.join(app.template_folder, 'predict_email.html')):
        return "Error: predict_email.html template not found.", 404
    return render_template("predict_email.html")

@app.route("/check_model")
def check_model():
    """API Endpoint: Checks model status."""
    # (Implementation unchanged)
    print("Checking model status...")
    model_exists = os.path.exists(MODEL_PATH)
    metrics_exists = os.path.exists(METRICS_PATH)
    model_filename = os.path.basename(MODEL_PATH) if model_exists else None
    metrics = {}
    model_type = None
    error_message = None

    if metrics_exists:
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
                model_type = metrics.get("model_type")
            print(f"Metrics loaded successfully from {METRICS_PATH}.")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading metrics file {METRICS_PATH}: {e}")
            error_message = f"Could not load metrics file: {e}"
    elif model_exists:
         print(f"Metrics file not found at {METRICS_PATH}, but model file exists at {MODEL_PATH}.")
    else:
        print("Neither model nor metrics file found.")

    response_data = {
        "model_exists": model_exists,
        "model_filename": model_filename,
        "model_metrics": metrics,
        "model_type": model_type
    }
    if error_message:
        response_data["error"] = error_message
        print(f"Returning model check status with error: {error_message}")
        # Return 200 OK but include the error in the response body
        return jsonify(response_data)

    print(f"Returning model check status: Exists={model_exists}, Metrics Found={metrics_exists}, Type={model_type}")
    return jsonify(response_data)


@app.route("/train_model", methods=["POST"])
def train_model():
    """API Endpoint (POST): Trains a classification model with detailed pre-processing."""
    print("Received request to train model.")
    start_time = time.time()

    if not GMAIL_API_AVAILABLE:
         print("Error: Gmail API module not available. Cannot train model.")
         return jsonify({"error": "Gmail API module not available. Cannot train model."}), 503 # 503 Service Unavailable

    selected_model_key = "random_forest"
    try:
        # Check if request has JSON data
        request_data = request.get_json()
        if request_data and 'selected_model' in request_data:
            selected_model_key = request_data['selected_model']
            if selected_model_key not in ["random_forest", "logistic_regression", "svm"]:
                print(f"Warning: Invalid model key '{selected_model_key}', defaulting to random_forest.")
                selected_model_key = "random_forest"
        elif request_data is None:
             print("No JSON data received in request. Defaulting to random_forest.")
        else:
             print("No 'selected_model' key in JSON data. Defaulting to random_forest.")
        print(f"Selected model type for training: {selected_model_key}")
    except Exception as e:
        print(f"Error parsing model selection from request: {e}. Defaulting to random_forest.")
        selected_model_key = "random_forest" # Ensure default on error

    print("Starting model training process...")
    try:
        # --- 1. Fetch Data ---
        print("Fetching emails...")
        service = get_gmail_service()
        if not service: return jsonify({"error": "Failed to authenticate with Gmail API."}), 500
        # Reduce max_results for faster testing if needed
        inbox_ids = list_messages(service, query='label:inbox category:primary', max_results=250) # Reduced for testing
        trash_ids = list_messages(service, query='label:trash', max_results=250) # Reduced for testing
        if not inbox_ids and not trash_ids: return jsonify({"error": "No emails found in primary INBOX or TRASH."}), 400
        inbox_emails = get_batch_message_details(service, [msg['id'] for msg in inbox_ids]) if inbox_ids else []
        trash_emails = get_batch_message_details(service, [msg['id'] for msg in trash_ids]) if trash_ids else []
        print(f"Fetched {len(inbox_emails)} INBOX and {len(trash_emails)} TRASH emails.")

        # Check for sufficient data *after* fetching details
        if len(inbox_emails) + len(trash_emails) < 10:
             return jsonify({"error": f"Insufficient total emails with details fetched ({len(inbox_emails) + len(trash_emails)} < 10)."}), 400
        if not inbox_emails or not trash_emails:
             print("WARNING: Training data might be imbalanced or missing one class.")
             # Allow training but warn, or return error if strictly required
             if len(set([0]*len(inbox_emails) + [1]*len(trash_emails))) < 2:
                  return jsonify({"error": "Training requires emails from both INBOX and TRASH to be successfully fetched."}), 400

        all_emails = inbox_emails + trash_emails
        labels = [0] * len(inbox_emails) + [1] * len(trash_emails)

        # --- 2. Preprocessing & Splitting ---
        print("Applying detailed pre-processing steps...")
        # Combine subject and snippet
        email_texts_raw = [f"{email.get('subject', '')} {email.get('snippet', '')}".strip() for email in all_emails]
        # Apply the detailed pre-processing function
        email_texts_processed = [preprocess_text_detailed(text) for text in email_texts_raw]

        # Optional: Print examples after processing
        # print("--- Example processed texts (Training) ---")
        # print(f"Processed 0: {email_texts_processed[0][:100]}...")
        # if len(labels) > len(inbox_emails): print(f"Processed spam: {email_texts_processed[len(inbox_emails)][:100]}...")
        # print("------------------------------------------")

        # Vectorization (Step 8 from previous context)
        print("Vectorizing and splitting data...")
        vectorizer = CountVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=False, # Already done in preprocess_text_detailed
            token_pattern=r"(?u)\b\w\w+\b|<URL>|<EMAIL>|<NUMBER>" # Handle placeholders
        )
        X = vectorizer.fit_transform(email_texts_processed) # Use processed text
        y = np.array(labels)

        # Check if X is empty after vectorization (can happen with very short/weird emails)
        if X.shape[0] == 0 or X.shape[1] == 0:
             return jsonify({"error": "Feature matrix X is empty after vectorization. Check email content and preprocessing."}), 400

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        print(f"Data shape: {X.shape}, Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
             return jsonify({"error": f"Train or test set is empty after split (Train: {X_train.shape[0]}, Test: {X_test.shape[0]}). Not enough diverse data?"}), 400

        # --- 3. Model Training ---
        print(f"Training model: {selected_model_key}...")
        model = None; model_type_str = "Unknown"
        if selected_model_key == "logistic_regression":
            model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000, solver='liblinear'); model_type_str = "LogisticRegression"
        elif selected_model_key == "svm":
            model = LinearSVC(random_state=42, class_weight='balanced', dual="auto", max_iter=2000); model_type_str = "LinearSVC (SVM)"
        else: # Default to Random Forest
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1); model_type_str = "RandomForestClassifier"
        model.fit(X_train, y_train)
        print(f"Model '{model_type_str}' trained.")

        # --- 4. Evaluation ---
        print("Evaluating model...")
        y_pred = model.predict(X_test); y_probs = None; roc_auc = None; fpr = None; tpr = None
        try:
             if hasattr(model, "predict_proba"):
                 y_probs = model.predict_proba(X_test)[:, 1]; fpr, tpr, _ = roc_curve(y_test, y_probs); roc_auc = auc(fpr, tpr)
             elif hasattr(model, "decision_function"):
                 y_scores = model.decision_function(X_test); fpr, tpr, _ = roc_curve(y_test, y_scores); roc_auc = auc(fpr, tpr)
             else:
                 print("Warning: Model does not support probability/decision scores. AUC not calculated.")
        except Exception as roc_err: print(f"Error calculating ROC/AUC: {roc_err}")
        accuracy = accuracy_score(y_test, y_pred); precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0); f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        print(f"Metrics: Acc={accuracy:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, AUC={roc_auc if roc_auc is not None else 'N/A'}")

        # --- 5. Plotting ---
        print("Generating ROC curve plot...")
        plt.figure(figsize=(8, 6));
        if fpr is not None and tpr is not None and roc_auc is not None:
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC ({model_type_str}, AUC = {roc_auc:.2f})')
        else:
            plt.text(0.5, 0.5, 'ROC Curve not available', ha='center', va='center', fontsize=12, color='red')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC Curve - {model_type_str}"); plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout();
        try:
            plt.savefig(PLOT_PATH); print(f"ROC curve saved to {PLOT_PATH}")
        except Exception as plot_err:
            print(f"Error saving ROC plot to {PLOT_PATH}: {plot_err}")
            # Don't fail the whole request, just log the error
        finally:
             plt.close(); # Ensure plot is closed even if saving fails

        # --- 6. Saving Model and Metrics ---
        print("Saving model and vectorizer...")
        try:
            # Save the vectorizer AND the model together
            with open(MODEL_PATH, "wb") as f: pickle.dump((vectorizer, model), f)
            print(f"Model and vectorizer saved to {MODEL_PATH}")
        except (IOError, pickle.PicklingError) as save_err:
            print(f"ERROR saving model/vectorizer to {MODEL_PATH}: {save_err}")
            traceback.print_exc()
            return jsonify({"error": "Critical error: Failed to save the trained model."}), 500

        metrics_data = {
            "model_type": model_type_str, "timestamp": time.time(),
            "accuracy": round(accuracy * 100, 2), "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2), "f1_score": round(f1 * 100, 2),
            "auc": round(roc_auc * 100, 2) if roc_auc is not None else None,
            "confusion_matrix": cm,
            "training_samples_total": len(all_emails),
            "training_samples_inbox": len(inbox_emails),
            "training_samples_trash": len(trash_emails),
            "test_samples": X_test.shape[0],
            "features_count": X.shape[1]
        }
        print(f"Metrics data: {json.dumps(metrics_data, indent=2)}")
        try:
            with open(METRICS_PATH, "w") as f: json.dump(metrics_data, f, indent=4)
            print(f"Metrics saved to {METRICS_PATH}")
        except IOError as save_err:
            print(f"ERROR saving metrics to {METRICS_PATH}: {save_err}")
            metrics_data["warning"] = "Failed to save metrics file." # Add warning to response

        end_time = time.time(); training_duration = end_time - start_time
        print(f"Training completed in {training_duration:.2f} seconds!")
        response_payload = {
            "message": f"Successfully trained and saved {model_type_str} model.",
            "training_duration_seconds": round(training_duration, 2),
            **metrics_data # Merge metrics into the response
        }
        return jsonify(response_payload)

    # --- Error Handling ---
    except FileNotFoundError as e:
        print(f"Error: File not found during training: {e}")
        traceback.print_exc()
        return jsonify({"error": f"File not found error: {e}."}), 500
    except HttpError as e:
        print(f"Google API Error during training: {e}")
        traceback.print_exc()
        error_details = f"Gmail API Error ({e.resp.status if hasattr(e, 'resp') else 'unknown'})"
        try:
            content = json.loads(e.content.decode('utf-8'))
            error_message = content.get('error', {}).get('message', 'No specific message.')
            error_details = f"{error_details}: {error_message}"
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError, Exception) as parse_err:
            print(f"Could not parse error details from HttpError content: {parse_err}")
        return jsonify({"error": error_details}), 500
    except MemoryError as e:
        print(f"Memory Error during training: {e}")
        traceback.print_exc()
        return jsonify({"error": "Memory Error encountered during training process. Consider reducing data size or feature count."}), 500
    except ValueError as e:
         print(f"Value Error during training: {e}")
         traceback.print_exc()
         return jsonify({"error": f"Value error during training: {str(e)}"}), 400 # Bad Request
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred during training: {str(e)}"}), 500


@app.route("/get_plot")
def get_plot():
    """API Endpoint: Sends the ROC curve plot image."""
    # (Implementation unchanged)
    print(f"Request received for plot file: {PLOT_PATH}")
    if os.path.exists(PLOT_PATH):
        try:
            return send_file(PLOT_PATH, mimetype='image/png', as_attachment=False)
        except Exception as e:
            print(f"Error sending plot file {PLOT_PATH}: {e}")
            return jsonify({"error": "Could not send plot file."}), 500
    else:
        print(f"Plot file not found at {PLOT_PATH}")
        return jsonify({"error": "Plot file not found. Train a model first."}), 404

# --- Email Processing and Confirmation API Routes ---

@app.route("/api/process_unread_emails", methods=["POST"])
def process_unread_emails_api():
    """API Endpoint (POST): Fetches, predicts, and processes unread emails."""
    print("API call received: /api/process_unread_emails")
    start_time = time.time()

    # --- 1. Pre-checks ---
    if not GMAIL_API_AVAILABLE: return jsonify({"error": "Gmail API module is not available."}), 503
    vectorizer, model = load_model_and_vectorizer()
    if not vectorizer or not model: return jsonify({"error": "Model or vectorizer not found. Please train a model first."}), 404

    # --- 2. Initialize ---
    processed_count = 0; auto_deleted_emails = []; auto_kept_emails = []; needs_confirmation = []; errors = []
    model_can_predict_proba = hasattr(model, "predict_proba")

    # --- 3. Get Gmail Service ---
    try:
        service = get_gmail_service()
        if not service: return jsonify({"error": "Failed to authenticate with Gmail API."}), 500
    except Exception as e: print(f"Error getting Gmail service: {e}"); traceback.print_exc(); return jsonify({"error": f"Failed to initialize Gmail service: {str(e)}"}), 500

    # --- 4. Fetch Unread Emails ---
    try:
        print(f"Fetching up to {PROCESS_EMAIL_LIMIT} unread emails...")
        query = 'is:unread in:inbox category:primary' # Focus on primary inbox unread
        message_items = list_messages(service, query=query, max_results=PROCESS_EMAIL_LIMIT)
        if not message_items: print("No unread emails found matching query."); return jsonify({"message": "Không tìm thấy email chưa đọc nào trong Hộp thư chính.", "processed_count": 0, "auto_deleted_emails": [], "auto_kept_emails": [], "needs_confirmation": [], "errors": []})
        message_ids = [msg['id'] for msg in message_items]
        print(f"Found {len(message_ids)} unread emails. Fetching details...")
        emails = get_batch_message_details(service, message_ids)
        print(f"Fetched details for {len(emails)} emails.")
        if not emails: # Handle case where details couldn't be fetched
             print("Warning: Could not fetch details for any found email IDs.")
             return jsonify({"message": "Tìm thấy ID email chưa đọc nhưng không thể lấy chi tiết.", "processed_count": 0, "auto_deleted_emails": [], "auto_kept_emails": [], "needs_confirmation": [], "errors": [{"id": "N/A", "subject": "Fetching Error", "error": "Không thể lấy chi tiết email."}]})
        processed_count = len(emails)
    except HttpError as e:
        print(f"Gmail API error fetching emails: {e}")
        traceback.print_exc()
        error_details = f"Gmail API Error ({e.resp.status if hasattr(e, 'resp') else 'unknown'})"
        try:
            content = json.loads(e.content.decode('utf-8'))
            error_message = content.get('error', {}).get('message', 'No specific message.')
            error_details = f"{error_details}: {error_message}"
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError, Exception) as parse_err:
             print(f"Could not parse error details from HttpError content: {parse_err}")
        return jsonify({"error": f"Lỗi khi lấy email: {error_details}"}), 500
    except Exception as e: print(f"Unexpected error fetching emails: {e}"); traceback.print_exc(); return jsonify({"error": f"Lỗi không mong muốn khi lấy email: {str(e)}"}), 500

    # --- 5. Preprocess and Predict Emails ---
    print(f"Preprocessing and predicting {len(emails)} emails...")
    # Combine subject and snippet
    email_texts_raw = [f"{email.get('subject', '')} {email.get('snippet', '')}".strip() for email in emails]
    # Apply the detailed pre-processing function
    email_texts_processed = [preprocess_text_detailed(text) for text in email_texts_raw]

    try:
        # Use the loaded vectorizer to transform the processed text
        X_new = vectorizer.transform(email_texts_processed)
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new) if model_can_predict_proba else None
        if probabilities is not None: print("Probabilities obtained.")
        else: print("Confidence thresholding disabled (model cannot predict probabilities).")
    except Exception as e:
        print(f"Error during model prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Lỗi khi dự đoán bằng model: {str(e)}"}), 500

    # --- 6. Categorize and Perform Actions ---
    print("Categorizing emails and performing actions...")
    for i, email in enumerate(emails):
        email_id = email['id']; subject = email.get('subject', '(Không có tiêu đề)'); snippet = email.get('snippet', '')
        predicted_action = predictions[i] # 0 for keep (inbox), 1 for delete (trash)
        email_data_for_response = {"id": email_id, "subject": subject, "snippet": snippet, "confidence_keep": None, "confidence_delete": None}
        action_performed = False # Flag to track if an action was attempted

        try:
            if probabilities is not None:
                # Assuming class 0 is 'keep' (inbox) and class 1 is 'delete' (trash)
                # Check model.classes_ if unsure about the order
                prob_keep = probabilities[i][0] # Probability of class 0
                prob_delete = probabilities[i][1] # Probability of class 1
                email_data_for_response["confidence_keep"] = round(prob_keep * 100, 1)
                email_data_for_response["confidence_delete"] = round(prob_delete * 100, 1)

                # Auto-delete if prediction is 'delete' and confidence is high
                if predicted_action == 1 and prob_delete >= CONFIDENCE_THRESHOLD:
                    print(f"Auto-deleting {email_id} (Conf: {prob_delete:.2f})")
                    action_performed = True
                    if delete_message(service, email_id): auto_deleted_emails.append(email_data_for_response)
                    else: errors.append({"id": email_id, "subject": subject, "error": "Lỗi khi tự động xóa (move to trash)"})

                # Auto-keep if prediction is 'keep' and confidence is high
                elif predicted_action == 0 and prob_keep >= CONFIDENCE_THRESHOLD:
                    print(f"Auto-keeping {email_id} (Conf: {prob_keep:.2f})")
                    action_performed = True
                    if mark_as_read(service, email_id): auto_kept_emails.append(email_data_for_response)
                    else: errors.append({"id": email_id, "subject": subject, "error": "Lỗi khi tự động giữ lại (đánh dấu đã đọc)"})

                # Otherwise, needs confirmation
                else:
                    print(f"Needs confirmation {email_id} (Keep: {prob_keep:.2f}, Del: {prob_delete:.2f})")
                    needs_confirmation.append(email_data_for_response)
            else: # No probabilities, always needs confirmation
                print(f"Needs confirmation {email_id} (no probability available)")
                needs_confirmation.append(email_data_for_response)

        except HttpError as api_err:
            print(f"API error processing {email_id}: {api_err}")
            traceback.print_exc()
            error_details = f"Gmail API Error ({api_err.resp.status if hasattr(api_err, 'resp') else 'unknown'})"
            try:
                content = json.loads(api_err.content.decode('utf-8'))
                error_message = content.get('error', {}).get('message', 'No specific message.')
                error_details = f"{error_details}: {error_message}"
            except (json.JSONDecodeError, AttributeError, UnicodeDecodeError, Exception) as parse_err:
                 print(f"Could not parse error details from HttpError content: {parse_err}")
            # Add error only if an action was attempted but failed due to API error
            if action_performed:
                 errors.append({"id": email_id, "subject": subject, "error": f"Lỗi API khi thực hiện hành động: {error_details}"})
            else: # If error occurred before action (e.g., during probability check?), add to needs confirmation? Or just log?
                 print(f"Non-action API error for {email_id}, adding to needs confirmation.")
                 needs_confirmation.append(email_data_for_response) # Add to confirmation if API error prevented auto-action

        except Exception as action_err:
            print(f"Unexpected error processing {email_id}: {action_err}")
            traceback.print_exc()
            # Add error only if an action was attempted but failed
            if action_performed:
                 errors.append({"id": email_id, "subject": subject, "error": f"Lỗi không mong muốn khi thực hiện hành động: {str(action_err)}"})
            else:
                 print(f"Non-action unexpected error for {email_id}, adding to needs confirmation.")
                 needs_confirmation.append(email_data_for_response) # Add to confirmation if error prevented auto-action


    # --- 7. Prepare and Return Response ---
    end_time = time.time(); duration = end_time - start_time
    print(f"Processing finished in {duration:.2f}s.")
    print(f"Summary: AutoDel={len(auto_deleted_emails)}, AutoKeep={len(auto_kept_emails)}, NeedsConf={len(needs_confirmation)}, Errors={len(errors)}")
    response_payload = {
        "message": f"Đã xử lý {processed_count} email trong {duration:.1f} giây.",
        "processed_count": processed_count, "auto_deleted_emails": auto_deleted_emails,
        "auto_kept_emails": auto_kept_emails, "needs_confirmation": needs_confirmation, "errors": errors
    }
    return jsonify(response_payload)


@app.route("/api/confirm_action", methods=["POST"])
def confirm_action_api():
    """
    API Endpoint (POST): Handles user confirmation actions (Keep/Delete).
    """
    print("API call received: /api/confirm_action")
    # --- 1. Pre-checks and Input Validation ---
    if not GMAIL_API_AVAILABLE:
        return jsonify({"error": "Gmail API module is not available."}), 503

    try:
        data = request.get_json()
        if not data: return jsonify({"error": "Yêu cầu không chứa dữ liệu JSON."}), 400

        email_id = data.get('email_id')
        action = data.get('action') # 'keep' or 'delete'
        subject = data.get('subject', '(Không có tiêu đề)') # Optional subject

        if not email_id or not action or action not in ['keep', 'delete']:
            return jsonify({"error": "Thiếu email_id hoặc action không hợp lệ ('keep' hoặc 'delete')."}), 400

    except Exception as e:
        print(f"Error parsing request data: {e}")
        return jsonify({"error": f"Lỗi xử lý dữ liệu yêu cầu: {str(e)}"}), 400

    # --- 2. Get Gmail Service ---
    try:
        service = get_gmail_service()
        if not service:
            return jsonify({"error": "Failed to authenticate with Gmail API."}), 500
    except Exception as e:
        print(f"Error getting Gmail service: {e}")
        return jsonify({"error": f"Failed to initialize Gmail service: {str(e)}"}), 500

    # --- 3. Perform Gmail Action ---
    action_successful = False
    error_message_detail = "Unknown error" # Default error message
    try:
        if action == 'delete':
            print(f"Performing action 'delete' for email {email_id}")
            action_successful = delete_message(service, email_id)
            if not action_successful: error_message_detail = "Gmail delete function failed."
        elif action == 'keep':
            print(f"Performing action 'keep' (mark as read) for email {email_id}")
            action_successful = mark_as_read(service, email_id) # Mark as read
            if not action_successful: error_message_detail = "Gmail mark as read function failed."

        if not action_successful:
            print(f"Action '{action}' failed for {email_id}. Detail: {error_message_detail}")
            return jsonify({"error": f"Hành động '{action}' trên Gmail không thành công cho email {email_id}."}), 500

    except HttpError as api_err:
        print(f"Gmail API error performing action '{action}' for {email_id}: {api_err}")
        traceback.print_exc()
        error_details = f"Gmail API Error ({api_err.resp.status if hasattr(api_err, 'resp') else 'unknown'})"
        try:
            content = json.loads(api_err.content.decode('utf-8'))
            error_msg_from_api = content.get('error', {}).get('message', 'No specific message.')
            error_details = f"{error_details}: {error_msg_from_api}"
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError, Exception) as parse_err:
             print(f"Could not parse error details from HttpError content: {parse_err}")
        return jsonify({"error": f"Lỗi API khi thực hiện hành động '{action}': {error_details}"}), 500
    except Exception as e:
        print(f"Unexpected error performing action '{action}' for {email_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Lỗi không mong muốn khi thực hiện hành động '{action}': {str(e)}"}), 500

    # --- 4. Log Feedback (Optional) ---
    # Log only if the action was successful on Gmail's side
    log_rl_feedback(email_id, subject, None, None, action)

    # --- 5. Return Success Response ---
    print(f"Action '{action}' confirmed successfully for email {email_id}.")
    return jsonify({"message": f"Đã xác nhận hành động '{action}' thành công cho email {email_id}."})


@app.route("/api/restore_email", methods=["POST"])
def restore_email_api():
    """
    API Endpoint (POST): Handles restoring an email from the Trash (untrash).
    """
    print("API call received: /api/restore_email")
    # --- 1. Pre-checks and Input Validation ---
    if not GMAIL_API_AVAILABLE:
        return jsonify({"error": "Gmail API module is not available."}), 503

    try:
        data = request.get_json()
        if not data: return jsonify({"error": "Yêu cầu không chứa dữ liệu JSON."}), 400
        email_id = data.get('email_id')
        if not email_id: return jsonify({"error": "Thiếu email_id trong yêu cầu."}), 400

    except Exception as e:
        print(f"Error parsing request data: {e}")
        return jsonify({"error": f"Lỗi xử lý dữ liệu yêu cầu: {str(e)}"}), 400

    # --- 2. Get Gmail Service ---
    try:
        service = get_gmail_service()
        if not service: return jsonify({"error": "Failed to authenticate with Gmail API."}), 500
    except Exception as e: print(f"Error getting Gmail service: {e}"); return jsonify({"error": f"Failed to initialize Gmail service: {str(e)}"}), 500

    # --- 3. Perform Untrash Action ---
    action_successful = False
    error_message_detail = "Unknown error during untrash"
    try:
        print(f"Performing action 'untrash' for email {email_id}")
        # Move back to INBOX and mark as UNREAD
        action_successful = modify_message_labels(service, email_id,
                                                  labels_to_add=['INBOX', 'UNREAD'],
                                                  labels_to_remove=['TRASH'])
        if not action_successful: error_message_detail = "Gmail modify labels function failed for untrash."


        if not action_successful:
            print(f"Action 'untrash' failed for {email_id}. Detail: {error_message_detail}")
            return jsonify({"error": f"Hành động 'untrash' trên Gmail không thành công cho email {email_id}."}), 500

    except HttpError as api_err:
        print(f"Gmail API error performing 'untrash' for {email_id}: {api_err}")
        traceback.print_exc()
        error_details = f"Gmail API Error ({api_err.resp.status if hasattr(api_err, 'resp') else 'unknown'})"
        try:
            content = json.loads(api_err.content.decode('utf-8'))
            error_msg_from_api = content.get('error', {}).get('message', 'No specific message.')
            error_details = f"{error_details}: {error_msg_from_api}"
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError, Exception) as parse_err:
             print(f"Could not parse error details from HttpError content: {parse_err}")
        return jsonify({"error": f"Lỗi API khi thực hiện 'untrash': {error_details}"}), 500
    except Exception as e:
        print(f"Unexpected error performing 'untrash' for {email_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Lỗi không mong muốn khi thực hiện 'untrash': {str(e)}"}), 500

    # --- 4. Return Success Response ---
    print(f"Action 'untrash' completed successfully for email {email_id}.")
    return jsonify({"message": f"Đã khôi phục email {email_id} thành công."})


# --- Old/Placeholder Route ---
@app.route("/api/predict", methods=["POST"])
def predict_api_manual():
    """Placeholder/Old route - No longer primary prediction method."""
    print("Placeholder route /api/predict called.")
    # You might want to implement manual prediction based on text input here later
    # For now, return 'Not Implemented'
    return jsonify({"message": "Manual prediction endpoint - Not currently implemented. Use /api/process_unread_emails."}), 501

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Flask Application ---")
    print(f" * Base Directory: {BASE_DIR}")
    print(f" * Model Path: {MODEL_PATH}")
    print(f" * Metrics Path: {METRICS_PATH}")
    print(f" * Plot Path: {PLOT_PATH}")
    print(f" * RL Feedback Log: {RL_FEEDBACK_LOG}")
    print(f" * Gmail API Available: {GMAIL_API_AVAILABLE}")
    print(f" * Email Processing Limit: {PROCESS_EMAIL_LIMIT}")
    print(f" * Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    # Set debug=False for production!
    # Use host='0.0.0.0' to make accessible on network
    print(f" * Flask Debug Mode: True (Set to False in production!)")
    app.run(debug=True, host='0.0.0.0', port=5000)
