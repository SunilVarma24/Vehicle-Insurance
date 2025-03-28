# src/model.py
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_xgboost(x_train, y_train, scale_pos_weight=None, random_state: int = 44):
    """Train an XGBoost classifier using class weight (if provided)."""
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='aucpr', random_state=random_state)
    model.fit(x_train, y_train)
    return model

def balanced_train_xgboost(x_train, y_train, random_state: int = 44):
    """Train an XGBoost classifier on a balanced dataset."""
    model = XGBClassifier(use_label_encoder=False, eval_metric='aucpr', random_state=random_state)
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x, y, dataset_name=""):
    """Evaluate the model and print confusion matrix and classification report."""
    y_pred = model.predict(x)
    print(f"{dataset_name} Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(y, y_pred))