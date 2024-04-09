
from sklearn.linear_model import LinearRegression
"""Module providing a linear models."""

import pandas as pd
"""Module providing framework for reading and organizing data."""

# Load the dataset (replace this with your dataset)
data = pd.read_csv("house_prices.csv")
print(data.head(5))

# Split the data into features (X) and target variable (y)
'''
        |    /
 Price  |  _/
        | /
        |/____________
            Features
'''

# Drop unused columns
X_unused =  [
       'price',
       'mainroad',
       'guestroom',
       'basement',
       'hotwaterheating',
       'airconditioning',
       'parking',
       'prefarea',
       'furnishingstatus'
]
X = data.drop(columns=X_unused)
y = data['price']

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# House specs: [ area, bedrooms, bathrooms, stories ]
new_houses = [
       [2500, 3, 2, 1],
       [5500, 6, 5, 2],
       [1200, 3, 1.5, 1],
       [3000, 4, 4, 2],
       [1000, 2, 2, 1],
]
predicted_prices = model.predict(new_houses)
print("\nPredicted price for the new houses: ")
for house in predicted_prices:
       print(f"\t{house}")

