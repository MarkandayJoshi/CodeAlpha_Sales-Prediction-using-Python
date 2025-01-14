import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare the data
def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Drop the unnamed index column if it exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

# Create visualization for correlation analysis
def plot_correlation_heatmap(df):
    """Plot a heatmap to visualize correlations between features."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Advertising Channels vs Sales')
    plt.show()

# Create scatter plots for each feature vs Sales
def plot_scatter_plots(df):
    """Create scatter plots for each feature against Sales."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    features = ['TV', 'Radio', 'Newspaper']
    
    for i, feature in enumerate(features):
        sns.scatterplot(data=df, x=feature, y='Sales', ax=axes[i])
        axes[i].set_title(f'{feature} vs Sales')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Sales')
    
    plt.tight_layout()
    plt.show()

# Train the model and make predictions
def train_model(X, y):
    """Train a linear regression model and return the model and predictions."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred

# Evaluate the model
def evaluate_model(model, X_test, y_test, y_pred):
    """Evaluate the model's performance and print metrics."""
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\nModel Performance Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    # Print feature coefficients
    feature_coefficients = pd.DataFrame({
        'Feature': ['TV', 'Radio', 'Newspaper'],
        'Coefficient': model.coef_
    })
    print("\nFeature Coefficients:")
    print(feature_coefficients)

def predict_sales(model, tv_budget, radio_budget, newspaper_budget):
    """Predict sales based on advertising budgets."""
    prediction = model.predict([[tv_budget, radio_budget, newspaper_budget]])
    return prediction[0]

def main():
    # Load the data
    df = load_data('Advertising.csv')
    if df is None:
        return  # Exit if data loading failed
    
    print("Dataset Shape:", df.shape)
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Create visualizations
    plot_correlation_heatmap(df)
    plot_scatter_plots(df)
    
    # Prepare features and target
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    # Train and evaluate the model
    model, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)
    evaluate_model(model, X_test, y_test, y_pred)
    
    # Example prediction
    example_prediction = predict_sales(model, 150, 30, 40)
    print(f"\nExample Prediction:")
    print(f"For TV=$150k, Radio=$30k, Newspaper=$40k")
    print(f"Predicted Sales: ${example_prediction:.2f}k")

if __name__ == "__main__":
    main()
