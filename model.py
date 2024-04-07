from sklearn import LinearRegression
import pandas as pd

# Load the dataset (replace this with your dataset)
data = pd.read_csv("house_price_index.csv")

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['Price ($)'])
y = data['Price ($)']

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Example prediction for a new house
new_house = [[2500, 3, 2, 'Suburb']]  # Example features for a new house
predicted_price = model.predict(new_house)
print("Predicted price for the new house:", predicted_price[0])
