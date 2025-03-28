# src/eda.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_response_distribution(df):
    """Pie chart of target Response distribution."""
    response_counts = df["Response"].value_counts(normalize=True) * 100
    plt.figure(figsize=(6, 4))
    plt.pie(response_counts, labels=response_counts.index, autopct='%1.1f%%')
    plt.title("Distribution of Response")
    plt.axis('equal')
    plt.show()

def plot_gender_vs_response(df):
    """Bar chart of Gender vs Response."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Gender', hue='Response', palette='viridis')
    plt.title('Gender vs Response')
    plt.xlabel('Gender (1 = Male, 0 = Female)')
    plt.ylabel('Count')
    plt.legend(title='Response', labels=['No (0)', 'Yes (1)'])
    plt.show()

def plot_age_distribution(df):
    """Histogram of Age distribution by Response."""
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x='Age', hue='Response', kde=True, palette='husl', bins=30, multiple='stack')
    plt.title('Age Distribution by Response')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend(title='Response', labels=['Yes (1)', 'No (0)'])
    plt.show()

def plot_driving_license_vs_response(df):
    """Bar chart of Driving License vs Response."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Driving_License', hue='Response', palette='viridis')
    plt.title('Driving License vs Response')
    plt.xlabel('Driving License (1 = Yes, 0 = No)')
    plt.ylabel('Count')
    plt.legend(title='Response', labels=['No (0)', 'Yes (1)'])
    plt.show()

def plot_region_code_distribution(df):
    """Bar chart of top 10 Region Codes by count."""
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='Region_Code', palette='viridis', order=df['Region_Code'].value_counts().index[:10])
    plt.title('Count of Records by Region Code')
    plt.xlabel('Top 10 Region Code')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

def plot_previously_insured_vs_response(df):
    """Bar chart of Previously Insured vs Response."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Previously_Insured', hue='Response', palette='viridis')
    plt.title('Previously_Insured vs Response')
    plt.xlabel('Previously_Insured (1 = Yes, 0 = No)')
    plt.ylabel('Count')
    plt.legend(title='Response', labels=['No (0)', 'Yes (1)'])
    plt.show()

def plot_vehicle_age_vs_response(df):
    """Bar chart of Vehicle Age vs Response."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Vehicle_Age', hue='Response', palette='viridis')
    plt.title('Vehicle_Age vs Response')
    plt.xlabel('Vehicle_Age (< 1 Year = 0, 1-2 Year = 1, > 2 Years)')
    plt.ylabel('Count')
    plt.legend(title='Response', labels=['No (0)', 'Yes (1)'])
    plt.show()

def plot_vehicle_damage_vs_response(df):
    """Bar chart of Vehicle Damage vs Response."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Vehicle_Damage', hue='Response', palette='viridis')
    plt.title('Vehicle_Damage vs Response')
    plt.xlabel('Vehicle_Damage (1 = Yes, 0 = No)')
    plt.ylabel('Count')
    plt.legend(title='Response', labels=['No (0)', 'Yes (1)'])
    plt.show()

def plot_annual_premium_distribution(df):
    """Histogram of Annual Premium distribution by Response."""
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x='Annual_Premium', hue='Response', kde=True, palette='husl', bins=30, multiple='stack')
    plt.title('Annual_Premium Distribution by Response')
    plt.xlabel('Annual_Premium')
    plt.ylabel('Count')
    plt.legend(title='Response', labels=['Yes (1)', 'No (0)'])
    plt.show()

def plot_policy_sales_channel_distribution(df):
    """Histogram of Policy Sales Channel distribution by Response."""
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x='Policy_Sales_Channel', hue='Response', kde=True, palette='husl', bins=30, multiple='stack')
    plt.title('Policy_Sales_Channel Distribution by Response')
    plt.xlabel('Policy_Sales_Channel')
    plt.ylabel('Count')
    plt.legend(title='Response', labels=['Yes (1)', 'No (0)'])
    plt.show()

def plot_correlation_heatmap(df):
    """Heatmap of correlations between numeric columns."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='crest')
    plt.show()

def plot_pca(x, y):
    """Scatter plot of PCA components of the feature space."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(6, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label='Class 0', alpha=0.5)
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label='Class 1', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('PCA of Feature Space')
    plt.show()