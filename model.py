import pandas as pd # Module providing framework for reading and organizing data.
from pandas import DataFrame
from sklearn.linear_model import LinearRegression # Module providing a regression models to fit data
from sklearn.linear_model import LogisticRegression
import asyncio
import time
import locale
import matplotlib as plt

class House:
       def __init__(self, 
              sqft: float,
              beds: int,
              baths: float,
              stories: int,
       ): 
              self.sqft = sqft 
              self.beds = beds
              self.baths = baths
              self.stories = stories
              
       def get_house_data_list(self) -> list:
              data = list()
              data.append(self.sqft)
              data.append(self.beds)
              data.append(self.baths)
              data.append(self.stories)
              return data

def load_data(csv_file: str) -> DataFrame:
       ''' Load data from a .csv file into a Pandas DataFrame'''
        # Load the dataset (replace this with your dataset)
       data = pd.read_csv(csv_file)
       print("\t Loaded Data\t")
       print(data.head(5))
       return data

async def train_model(X, y) -> LinearRegression:
       # Initialize and train the linear regression model
       model = LinearRegression()
       model.fit(X, y)
       print(f"Model fit complete")
       return model

def print_prices(houses, prices):

       locale.setlocale(locale.LC_ALL, '')  # Use '' for auto, or specify a locale like 'en_US.UTF-8'
       print("\n[Sqft,bed,bath,floors]\t\tPrice")
       print("--------------------------------------------")
       for house, price in zip(houses, prices):
              formatted_number = locale.format_string("%d", price, grouping=True)
              print(f'{house}\t  {formatted_number}')
              
       print("\n")
       return 


async def main():
       start_time = time.time()
       data = load_data("house_prices.csv")
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
       
       print("Training model...")
       
       model = await train_model(X, y)

       # House specs: [ area, bedrooms, bathrooms, stories ]
       new_houses = [
              [2500, 3, 2, 1],
              [5500, 6, 5, 2],
              [1200, 3, 1.5, 1],
              [3000, 4, 4, 2],
              [1000, 2, 2, 1],
       ]
       
       house1 = House(1800, 3, 1.5, 1)
       new_houses.append(house1.get_house_data_list())
       print(new_houses)
       predicted_prices = model.predict(new_houses)
       
       
       # print("\n[Sqft,bed,bath,floors]\t\tPrice")
       # print("--------------------------------------------")
       # for house, price in zip(new_houses, predicted_prices):
       #        print(f"{house}:\t\t$ {float(price):0.2f}")
       # print("\n")
       print(print_prices(new_houses, predicted_prices))
       end_time = time.time()
       

       execution_time = end_time - start_time
       print(f"Execution time: {execution_time} seconds\n")

if __name__ == "__main__":
       asyncio.run(main())
