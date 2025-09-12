# House Price Prediction

This project predicts house prices using a linear regression model trained on real estate data. It utilizes Python, pandas, and scikit-learn to process data and build the model.

## Features

- Loads and processes house price data from CSV files
- Trains a linear regression model to predict prices based on features like area, bedrooms, bathrooms, and stories
- Predicts prices for new house specifications
- Displays results in a readable format

## Files

- `model.py`: Main script for data loading, model training, and prediction
- `house_prices.csv`: Dataset containing house features and prices
- `requirements.txt`: Lists required Python packages (`pandas`, `scikit-learn`)
- `house_price_index.csv`: (Optional) Additional data for analysis

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the model:**
   ```bash
   python model.py
   ```

3. **View predictions:**  
   The script will output predicted prices for sample house specifications.

## Data

The dataset (`house_prices.csv`) includes columns:
- `price`, `area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, `furnishingstatus`

## Requirements

- Python 3.7+
- pandas
- scikit-learn

## License

See `LICENSE` for details.
