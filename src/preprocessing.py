# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek

def split_features_target(df: pd.DataFrame):
    """Split DataFrame into independent features (x) and target (y)."""
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return x, y

def calculate_class_weight(y: pd.Series) -> float:
    """Calculate class weight based on imbalance in target."""
    num_negatives = (y == 0).sum()
    num_positives = (y == 1).sum()
    scale_pos_weight = num_negatives / num_positives
    print(f"Class Weightage (scale_pos_weight): {scale_pos_weight:.2f}")
    return scale_pos_weight

def train_val_test_split(x: pd.DataFrame, y: pd.Series, test_size: float = 0.15, val_size: float = 0.05, random_state: int = 44):
    """Split the data into train, validation, and test sets."""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=random_state, shuffle=True, stratify=y_train)
    print("Training:", x_train.shape, y_train.shape)
    print("Testing:", x_test.shape, y_test.shape)
    print("Validation:", x_val.shape, y_val.shape)
    return x_train, x_val, x_test, y_train, y_val, y_test

def scale_features(x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame, cols: list):
    """Scale specified columns in the dataset using StandardScaler."""
    scaler = StandardScaler()
    x_train[cols] = scaler.fit_transform(x_train[cols])
    x_train = pd.DataFrame(x_train)
    
    x_val[cols] = scaler.transform(x_val[cols])
    x_val = pd.DataFrame(x_val)
    
    x_test[cols] = scaler.transform(x_test[cols])
    x_test = pd.DataFrame(x_test)
    
    return x_train, x_val, x_test

def balance_dataset(x: pd.DataFrame, y: pd.Series):
    """Balance the dataset using EditedNearestNeighbours and SMOTETomek."""
    
    enn = EditedNearestNeighbours(n_neighbors=5, n_jobs=-1)
    X_resampled, y_resampled = enn.fit_resample(x, y)
    smote_t = SMOTETomek(n_jobs=-1)
    X_resampled_new, y_resampled_new = smote_t.fit_resample(X_resampled, y_resampled)
    return X_resampled_new, y_resampled_new