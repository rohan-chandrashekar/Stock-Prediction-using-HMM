import logging
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas_datareader import data
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import argparse
import sys
from typing import Optional, Tuple, List

class HMMStockPredictor:
    """
    Hidden Markov Model-based stock price predictor.
    """
    def __init__(self, company: str, start_date: str, end_date: str, future_days: int, train_size: float = 0.8, n_hidden_states: int = 4, n_latency_days: int = 10, n_intervals_frac_change: int = 50, n_intervals_frac_high: int = 10, n_intervals_frac_low: int = 10, verbose: bool = False):
        self._init_logger(verbose)
        self.company = company
        self.start_date = start_date
        self.end_date = end_date
        self.n_latency_days = n_latency_days
        self.hmm = GaussianHMM(n_components=n_hidden_states)
        self._split_train_test_data(train_size)
        self._compute_all_possible_outcomes(
            n_intervals_frac_change, n_intervals_frac_high, n_intervals_frac_low)
        self.predicted_close = None
        self.days_in_future = future_days

    def _init_logger(self, verbose: bool = False):
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    def _split_train_test_data(self, test_size: float):
        try:
            used_data = data.DataReader(
                self.company, "yahoo", self.start_date, self.end_date)
        except Exception as e:
            self._logger.error(f"Error fetching data: {e}")
            sys.exit(1)
        _train_data, test_data = train_test_split(
            used_data, test_size=test_size, shuffle=False)
        self.train_data = _train_data.drop(["Volume", "Adj Close"], axis=1)
        self.test_data = test_data.drop(["Volume", "Adj Close"], axis=1)
        self.days = len(test_data)

    @staticmethod
    def _extract_features(data: pd.DataFrame) -> np.ndarray:
        open_price = np.array(data["Open"])
        close_price = np.array(data["Close"])
        high_price = np.array(data["High"])
        low_price = np.array(data["Low"])
        frac_change = (close_price - open_price) / open_price
        frac_high = (high_price - open_price) / open_price
        frac_low = (open_price - low_price) / open_price
        return np.column_stack((frac_change, frac_high, frac_low))

    def fit(self):
        """Fit the HMM model to the training data."""
        observations = HMMStockPredictor._extract_features(self.train_data)
        self.hmm.fit(observations)

    def _compute_all_possible_outcomes(self, n_intervals_frac_change: int, n_intervals_frac_high: int, n_intervals_frac_low: int):
        frac_change_range = np.linspace(-0.1, 0.1, n_intervals_frac_change)
        frac_high_range = np.linspace(0, 0.1, n_intervals_frac_high)
        frac_low_range = np.linspace(0, 0.1, n_intervals_frac_low)
        self._possible_outcomes = np.array(list(itertools.product(
            frac_change_range, frac_high_range, frac_low_range)))

    def _get_most_probable_outcome(self, day_index: int) -> np.ndarray:
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = self.test_data.iloc[previous_data_start_index:previous_data_end_index]
        previous_data_features = HMMStockPredictor._extract_features(previous_data)
        outcome_score = []
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack(
                (previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))
        most_probable_outcome = self._possible_outcomes[np.argmax(outcome_score)]
        return most_probable_outcome

    def predict_close_price(self, day_index: int) -> float:
        open_price = self.test_data.iloc[day_index]["Open"]
        (
            predicted_frac_change,
            pred_frac_high,
            pred_frac_low,
        ) = self._get_most_probable_outcome(day_index)
        return open_price * (1 + predicted_frac_change)

    def predict_close_prices_for_period(self) -> List[float]:
        predicted_close_prices = []
        for day_index in tqdm(range(self.days), desc="Predicting period close prices"):
            predicted_close_prices.append(self.predict_close_price(day_index))
        self.predicted_close = predicted_close_prices
        return predicted_close_prices

    def real_close_prices(self) -> pd.DataFrame:
        actual_close_prices = self.test_data.loc[:, ["Close"]]
        return actual_close_prices

    def add_future_days(self):
        last_day = self.test_data.index[-1] + \
            timedelta(days=self.days_in_future)
        future_dates = pd.date_range(
            self.test_data.index[-1] + pd.offsets.DateOffset(1), last_day)
        second_df = pd.DataFrame(index=future_dates, columns=["High", "Low", "Open", "Close"])
        self.test_data = pd.concat([self.test_data, second_df])
        self.test_data.iloc[self.days]["Open"] = self.test_data.iloc[self.days - 1]["Close"]

    def predict_close_price_fut_days(self, day_index: int) -> float:
        open_price = self.test_data.iloc[day_index]["Open"]
        (
            predicted_frac_change,
            pred_frac_high,
            pred_frac_low,
        ) = self._get_most_probable_outcome(day_index)
        predicted_close_price = open_price * (1 + predicted_frac_change)
        self.test_data.iloc[day_index]["Close"] = predicted_close_price
        self.test_data.iloc[day_index]["High"] = open_price * \
            (1 + pred_frac_high)
        self.test_data.iloc[day_index]["Low"] = open_price * \
            (1 - pred_frac_low)
        return predicted_close_price

    def predict_close_prices_for_future(self) -> List[float]:
        predicted_close_prices = []
        future_indices = len(self.test_data) - self.days_in_future
        print("Predicting future Close prices from " +str(self.test_data.index[future_indices]) + " to " + str(self.test_data.index[-1]))
        for day_index in tqdm(range(future_indices, len(self.test_data)), desc="Predicting future close prices"):
            predicted_close_prices.append(self.predict_close_price_fut_days(day_index))
            try:
                self.test_data.iloc[day_index +1]["Open"] = self.test_data.iloc[day_index]["Close"]
            except IndexError:
                continue
        self.predicted_close = predicted_close_prices
        return predicted_close_prices

def plot_results(in_df: pd.DataFrame, stock_name: str, save_path: Optional[str] = None):
    """Plot actual vs predicted close prices."""
    in_df = in_df.reset_index()
    ax = plt.gca()
    in_df.plot(kind="line", x="Date", y="Actual_Close", ax=ax)
    in_df.plot(kind="line", x="Date", y="Predicted_Close", color="red", ax=ax)
    plt.ylabel("Daily Close Price (in USD)")
    plt.title(str(stock_name) + " daily closing stock prices")
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close("all")

def calc_mse(input_df: pd.DataFrame, verbose: bool = True) -> float:
    """Calculate mean squared error and print accuracy."""
    actual_arr = (input_df.loc[:, "Actual_Close"]).values
    pred_arr = (input_df.loc[:, "Predicted_Close"]).values
    mse = mean_squared_error(actual_arr, pred_arr)
    if verbose:
        print("Mean Squared Error: " + str(mse))
    res = np.abs(actual_arr - pred_arr) / actual_arr * 100
    if verbose:
        print("Accuracy: ", 100 - np.mean(res))
    return mse

def export_to_excel(df: pd.DataFrame, filename: str):
    """Export DataFrame to Excel file."""
    df.to_excel(filename, index=False)

def use_stock_predictor(args):
    print(f"Using continuous Hidden Markov Models to predict stock prices for {args.company}")
    stock_predictor = HMMStockPredictor(
        company=args.company,
        start_date=args.start,
        end_date=args.end,
        future_days=args.days,
        n_hidden_states=args.n_hidden_states,
        n_latency_days=args.n_latency_days,
        verbose=args.verbose
    )
    print("Training data: ")
    stock_predictor.fit()
    predicted_close = stock_predictor.predict_close_prices_for_period()
    actual_close = stock_predictor.real_close_prices()
    actual_close["Predicted_Close"] = predicted_close
    output_df = actual_close.rename(columns={"Close": "Actual_Close"})
    mse = calc_mse(output_df, verbose=args.verbose)
    if args.save_excel:
        excel_path = f"Outputs/{args.company}_HMM_Prediction_{mse:.6f}.xlsx"
        export_to_excel(output_df, excel_path)
        print(f"Results exported to {excel_path}")
    plot_path = f"Outputs/{args.company}_results_plot.png" if args.save_plot else None
    plot_results(output_df, args.company, save_path=plot_path)
    if args.save_plot:
        print(f"Plot saved to {plot_path}")
    stock_predictor.add_future_days()
    future_pred_close = stock_predictor.predict_close_prices_for_future()
    print(f"The predicted stock prices for the next {args.days} days from {stock_predictor.end_date} are: ")
    for i in range(len(future_pred_close)):
        print(f"{stock_predictor.test_data.index[-args.days + i]} : {future_pred_close[i]}")

def main():
    parser = argparse.ArgumentParser(description="Stock Market Prediction using HMM")
    parser.add_argument('--company', type=str, required=True, help='Company ticker symbol (e.g., AAPL)')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=5, help='Number of days to predict into the future')
    parser.add_argument('--n_hidden_states', type=int, default=4, help='Number of hidden states for HMM')
    parser.add_argument('--n_latency_days', type=int, default=10, help='Number of latency days for prediction')
    parser.add_argument('--save_excel', action='store_true', help='Save results to Excel')
    parser.add_argument('--save_plot', action='store_true', help='Save plot to file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    use_stock_predictor(args)

if __name__ == "__main__":
    main()