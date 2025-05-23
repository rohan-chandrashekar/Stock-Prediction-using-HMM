# Stock Market Predictor using Hidden Markov Models (HMM)

## Overview 📊

This repository contains a stock market prediction tool utilizing **Hidden Markov Models (HMM)**, a robust statistical model, to predict future stock prices based on historical data. The project focuses on the analysis of Apple stocks as a case study, using **time series** data and the powerful capabilities of HMM to analyze patterns and dependencies in stock prices over time.

With financial markets being inherently unpredictable, this project seeks to provide an innovative approach to market prediction through **machine learning** models trained on historical stock data.

---

## Key Features 💡
- **Prediction of future stock prices** based on historical market data using the power of HMM.
- **Training and testing** on Apple stock prices for accurate predictions over a user-defined period.
- **Automated data collection** directly from Yahoo Finance using `pandas_datareader`.
- Calculation of **mean squared error (MSE)** between predicted and actual stock prices, with results optionally stored in Excel.
- Efficient **visualization of stock trends** with easy-to-interpret graphs for better understanding of price movements, with option to save plots.
- **Command-line interface** for flexible and scriptable usage.

---

## Dependencies 📦

To ensure a smooth run, install the following Python libraries (see `requirements.txt`):

- **pandas_datareader**
- **NumPy**
- **Matplotlib**
- **Hmmlearn**
- **Scikit-learn**
- **Tqdm**
- **Argparse**
- **Openpyxl** (for Excel export)

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

---

## How It Works 🚀

1. **Data Collection**: The tool fetches historical stock data from Yahoo Finance using `pandas_datareader` and splits it into training and testing sets.
   
2. **Model Training**: The fractional changes in stock prices (open, high, low, close) are used as observations to train the continuous Hidden Markov Model.
   
3. **Price Prediction**: Once trained, the model predicts stock prices for the next `N` days based on current data.

4. **Evaluation**: The predicted prices are compared with actual stock prices, and the Mean Squared Error (MSE) is calculated to assess the model's performance. 

5. **Visualization**: Results, including predicted and actual prices, are visualized using `matplotlib` for easy interpretation. Plots can be saved to disk.

6. **Export**: Optionally, results can be exported to Excel for further analysis.

---

## Project Structure 🗂

- **src/Stock_Analysis.py**: Main Python script that performs stock market prediction.
- **requirements.txt**: Lists all the dependencies required to run the project.
- **README.md**: This file, detailing how to use the project.
- **Outputs/**: Directory where plots and Excel files are saved.

---

## Usage 🖥️

To predict stock prices, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/rohan-chandrashekar/Stock-Market-HMM.git
   cd Stock-Market-HMM
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main Python script with command-line arguments:
   ```bash
   python src/Stock_Analysis.py --company AAPL --start 2022-01-01 --end 2023-01-01 --days 10 --save_excel --save_plot --verbose
   ```
   - `--company`: Ticker symbol (e.g., AAPL, MSFT, GOOGL)
   - `--start`: Start date (YYYY-MM-DD)
   - `--end`: End date (YYYY-MM-DD)
   - `--days`: Number of days to predict into the future (default: 5)
   - `--save_excel`: Save results to Excel (optional)
   - `--save_plot`: Save plot to file (optional)
   - `--verbose`: Verbose output (optional)
   - `--n_hidden_states`: Number of HMM hidden states (default: 4)
   - `--n_latency_days`: Number of latency days for prediction (default: 10)

---

## Example 📈

Here's an example of how the stock prices are predicted for 10 days and results are saved:

```bash
python src/Stock_Analysis.py --company AAPL --start 2022-01-01 --end 2023-01-01 --days 10 --save_excel --save_plot --verbose
```

- You'll receive an Excel file with predictions and the Mean Squared Error (MSE) score in the `Outputs/` directory.
- Plots will be saved in the `Outputs/` directory if `--save_plot` is used.

---

## Results and Evaluation 🎯

The project evaluates the predictions using **Mean Squared Error (MSE)** between the predicted and actual stock prices. All results can be saved in an Excel file for future reference. The tool provides **graphical outputs** that allow for easy analysis of the predictions against real-world stock data.

---

## Scope and Applications 🌐

This project demonstrates the powerful applications of **Hidden Markov Models** in **time-series analysis**, specifically for **financial markets**. HMMs are versatile and have numerous applications beyond stock prediction, such as:
- **Neuroscience**: Analyzing brain activity sequences.
- **Cryptanalysis**: Cracking codes with sequence patterns.
- **Machine Translation**: Predicting the likelihood of word sequences.
- **Gene Prediction and Virus Detection**: Identifying patterns in biological sequences.
