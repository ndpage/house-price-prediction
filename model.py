from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the dataset (replace this with your dataset)
data = pd.read_csv("house_prices.csv")

# Split the data into features (X) and target variable (y)
'''
        |    /
 Price  |  _/
        | /
        |/____________
            Features
'''

# area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus
X = data.drop(columns=['price','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus'])
y = data['price']

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Example prediction for a new house
# area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus
new_house = [[2500, 3, 2, 1]]  # Example features for a new house
predicted_price = model.predict(new_house)
print("Predicted price for the new house:", predicted_price[0])

