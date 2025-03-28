# main.py
from src.data import load_dataset, remove_outliers, feature_engineering
from src.eda import (
    plot_response_distribution,
    plot_gender_vs_response,
    plot_age_distribution,
    plot_driving_license_vs_response,
    plot_region_code_distribution,
    plot_previously_insured_vs_response,
    plot_vehicle_age_vs_response,
    plot_vehicle_damage_vs_response,
    plot_annual_premium_distribution,
    plot_policy_sales_channel_distribution,
    plot_correlation_heatmap,
    plot_pca
)
from src.preprocessing import (
    split_features_target,
    calculate_class_weight,
    train_val_test_split,
    scale_features,
    balance_dataset
)
from src.model import train_xgboost, evaluate_model, balanced_train_xgboost
import pandas as pd

def main():
    # 1. Load the dataset
    df = load_dataset(r"/content/vehicle insurance preprocessed.csv")
    print(df.head())

    # 2. Outliers Removal
    df_new = remove_outliers(df)

    # 3. Feature Engineering
    df_new = feature_engineering(df_new)

    # 4. Target variable value counts
    value_count = df_new.Response.value_counts()
    print('Class 0:', value_count[0])
    print('Class 1:', value_count[1])
    print('Proportion:', round(value_count[0] / value_count[1], 2), ': 1')
    print('Total Response:', len(value_count))

    # 5. EDA
    plot_response_distribution(df_new)
    plot_gender_vs_response(df_new)
    plot_age_distribution(df_new)
    plot_driving_license_vs_response(df_new)
    plot_region_code_distribution(df_new)
    plot_previously_insured_vs_response(df_new)
    plot_vehicle_age_vs_response(df_new)
    plot_vehicle_damage_vs_response(df_new)
    plot_annual_premium_distribution(df_new)
    plot_policy_sales_channel_distribution(df_new)
    plot_correlation_heatmap(df_new)
    
    # PCA: Split features and target for PCA visualization
    x, y = split_features_target(df_new)
    plot_pca(x, y)

    # 6. Independent and Dependent Features Split
    x, y = split_features_target(df_new)

    # 7. Calculate class weightage due to imbalance
    scale_pos_weight = calculate_class_weight(y)

    # 8. Train-test split
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)

    # 9. Feature Scaling
    cols = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
            'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
    
    # Scale the features using StandardScaler
    x_train, x_val, x_test = scale_features(x_train, x_val, x_test, cols)

    # 10. Model Training: Basic XGBoost Model Training with Class Weightage
    print("Basic XGBoost Model Training")
    model_basic = train_xgboost(x_train, y_train, scale_pos_weight=scale_pos_weight)
    print("Validation Evaluation:")
    evaluate_model(model_basic, x_val, y_val, dataset_name="Validation")
    print("Test Evaluation:")
    evaluate_model(model_basic, x_test, y_test, dataset_name="Test")

    # 11. XGBoost Model Training after balancing the dataset
    print("Balanced XGBoost Model Training after resampling")
    # Balance the dataset using under- and over-sampling techniques
    X_resampled, y_resampled = balance_dataset(x, y)
    # Perform train-test split on the resampled data
    x_train_b, x_val_b, x_test_b, y_train_b, y_val_b, y_test_b = train_val_test_split(pd.DataFrame(X_resampled), pd.Series(y_resampled))
    # Scale features for the balanced data
    x_train_b, x_val_b, x_test_b = scale_features(x_train_b, x_val_b, x_test_b, cols)
    model_balanced = balanced_train_xgboost(x_train_b, y_train_b)
    print("Validation Evaluation (Balanced):")
    evaluate_model(model_balanced, x_val_b, y_val_b, dataset_name="Validation")
    print("Test Evaluation (Balanced):")
    evaluate_model(model_balanced, x_test_b, y_test_b, dataset_name="Test")

if __name__ == "__main__":
    main()