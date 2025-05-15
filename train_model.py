import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from gmail_api import get_emails

def train_and_save_model(model_path):
    """Huấn luyện mô hình dự đoán email có bị xóa hay không"""

    # Lấy email từ cả hộp thư đến (INBOX) và thùng rác (TRASH)
    inbox_emails = get_emails("INBOX")  # Email hợp lệ (0)
    trash_emails = get_emails("TRASH")  # Email bị xóa (1)

    # Gộp tất cả email và tạo nhãn
    all_emails = inbox_emails + trash_emails
    subjects = [email["subject"] for email in all_emails]
    labels = [0] * len(inbox_emails) + [1] * len(trash_emails)  # 0 = không xóa, 1 = xóa

    # Xử lý dữ liệu
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(subjects)

    # Huấn luyện mô hình
    model = RandomForestClassifier()
    model.fit(X, labels)

    # Lưu mô hình & vectorizer
    with open(model_path, "wb") as f:
        pickle.dump((vectorizer, model), f)

    return model
