import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

# Load the data
filename = r'D:\Downloads\Env_Data.csv'
df = pd.read_csv(filename, header=0)

# Convert specific columns to float
columns_to_convert = ['temperature', 'humidity', 'wind_speed', 'air_quality_index', 'noise_level', 'precipitation', 'solar_radiation']
df[columns_to_convert] = df[columns_to_convert].astype(float)

# Display the DataFrame
print(df)

# Identify numeric columns
numerical_col = df._get_numeric_data()
print(numerical_col.columns)

# Handling Duplicates
duplicate_rows = df[df.duplicated()]
print(duplicate_rows)

# Check for missing values
print(df.isnull().sum())

# Handling Duplicates
df_no_duplicates = df.drop_duplicates()
print("\nDataFrame after removing duplicates:")
print(df_no_duplicates)

# Identify and remove outliers
outliers_removed_df = df.copy()

for column_name in df.columns:
    data_mean, data_std = df[column_name].mean(), df[column_name].std()

    # Define outliers
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off

    # Identify outliers
    outliers = df[(df[column_name] < lower) | (df[column_name] > upper)]
    print(f'Identified outliers for {column_name}: {len(outliers)}')

    outliers_removed_df = outliers_removed_df.reset_index(drop=True)
    outliers = outliers.reset_index(drop=True)

    # Remove rows with outliers
    outliers_removed_df = outliers_removed_df.drop(outliers.index)

# Print the DataFrame with outliers removed
print("\nDataFrame with Rows Containing Outliers Removed:")
print(outliers_removed_df)

# Understanding the shape of the dataset
print("\nShape of the dataset:")
print(df.shape)

# Column names
print("\nColumn names:")
print(df.columns)

# Distribution of each column
for c in df.columns:
    print(f'\nDistribution of {c}:')
    print(df[c].value_counts())

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())


# Column names
print("\nColumn names:")
print(df.columns)




class DataProcessor:
    def __init__(self, filename):
        self.df = pd.read_csv(filename, header=0)

    def handle_duplicates(self):
        self.df = self.df.drop_duplicates()

    def visualize_pairplot(self):
        sns.pairplot(self.df, vars=['temperature', 'humidity', 'wind_speed', 'air_quality_index'])
        plt.suptitle('Pair Plot of Numerical Features', y=1.02)
        plt.show()

    def visualize_correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()

    def visualize_box_plot(self):
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='month', y='air_quality_index', data=self.df, palette='rocket')
        plt.title('Box Plot of Air Quality Index Across Months')
        plt.show()

    def visualize_bar_chart(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(x='month', data=self.df, palette='viridis')
        plt.title('Distribution of Measurements Across Months')
        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.show()

    def visualize_histogram(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['temperature'], bins=20, color='salmon', edgecolor='black')
        plt.title('Distribution of Temperature')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Count')
        plt.show()
        
    def remove_outliers(self):
       outliers_removed_df = self.df.copy()
       for column_name in self.df.columns:
           data_mean, data_std = self.df[column_name].mean(), self.df[column_name].std()
           cut_off = data_std * 3
           lower, upper = data_mean - cut_off, data_mean + cut_off
           outliers = self.df[(self.df[column_name] < lower) | (self.df[column_name] > upper)]
           outliers_removed_df = outliers_removed_df.drop(outliers.index)
       return outliers_removed_df
        

class AirQualityModel:
    def __init__(self, df):
        self.df = df
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.X_test_scaled = None

    def prepare_data(self):
        # Assuming 'df' is your DataFrame with the provided columns
        self.X = self.df[['year', 'month', 'day', 'hour', 'temperature', 'humidity', 'wind_speed', 'noise_level', 'precipitation', 'solar_radiation']]
        self.y = self.df['air_quality_index']

        # Perform train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        # Apply Standard Scaling to selected numeric variables
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Fine-tune hyperparameters
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

        # Train the model on the scaled training data
        self.model.fit(X_train_scaled, self.y_train)

    def evaluate_model(self):
        # Make predictions on the scaled test set
        y_pred = self.model.predict(self.X_test_scaled)

        # Evaluate the model on the test set
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)

        print(f'Mean Absolute Error: {mae}')
        print(f'Mean Squared Error: {mse}')

        # Scatter plot for predicted vs actual values
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], color='red', linestyle='--', lw=2)
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()

    def save_model(self):
        # Apply Standard Scaling to the entire feature matrix
        X_scaled = self.scaler.fit_transform(self.X)

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, self.y, cv=5, scoring='neg_mean_squared_error')
        cv_scores_positive = -cv_scores

        # Print the cross-validated MSE scores
        print("Cross-Validated MSE Scores:", cv_scores_positive)
        print("Mean MSE:", cv_scores_positive.mean())

        # Save the trained model and scaler
        joblib.dump(self.model, 'air_quality_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')

if __name__ == '__main__':
    filename = r'D:\Downloads\Env_Data.csv'

    # Initialize DataProcessor
    data_processor = DataProcessor(filename)

    # Data Preprocessing Steps
    data_processor.handle_duplicates()
    outliers_removed_df = data_processor.remove_outliers()

    # Visualizations
    data_processor.visualize_pairplot()
    data_processor.visualize_correlation_heatmap()
    data_processor.visualize_box_plot()
    data_processor.visualize_histogram()

    # Initialize AirQualityModel
    air_quality_model = AirQualityModel(outliers_removed_df)

    # Model Preparation, Training, Evaluation, and Saving
    air_quality_model.prepare_data()
    air_quality_model.train_model()
    air_quality_model.evaluate_model()
    air_quality_model.save_model()

