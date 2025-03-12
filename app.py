import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load stock list from Excel file
@st.cache_data
def load_stocklist(file_path):
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    return xls, sheet_names

# Fetch stock data
def fetch_stock_data(tickers, period="1y"):
    data = yf.download(tickers, period=period)['Close']
    return data

# Portfolio Optimization using Modern Portfolio Theory (MPT)
def optimize_portfolio(data, risk_tolerance):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(data.columns)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    # Define objective function
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    if risk_tolerance == "Low":
        optimized = minimize(portfolio_volatility, weights, method="SLSQP", bounds=bounds, constraints=constraints)
    else:
        def neg_sharpe(weights):
            port_return = np.sum(mean_returns * weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -port_return / port_volatility  

        optimized = minimize(neg_sharpe, weights, method="SLSQP", bounds=bounds, constraints=constraints)

    return optimized.x, mean_returns, cov_matrix

# Streamlit UI
st.title("📈 Quantitative Stock Selection Model")

# Load stock data
file_path = "stocklist.xlsx"
xls, sheet_names = load_stocklist(file_path)

# User selects the sheet
sheet_selected = st.selectbox("Select Stock List Sheet:", sheet_names)
stock_df = pd.read_excel(xls, sheet_name=sheet_selected)
tickers = stock_df.iloc[:, 0].dropna().tolist()

# User Inputs
model = st.selectbox("Select a Quantitative Model:", ["Modern Portfolio Theory (MPT)", "Momentum Strategy"])
risk_tolerance = st.selectbox("Select Risk Tolerance:", ["Low", "Medium", "High"])
investment_horizon = st.selectbox("Investment Horizon:", ["Short-term", "Long-term"])
factors = st.multiselect("Factor Preferences:", ["Volatility", "Momentum", "Value", "Growth"])

if st.button("Generate Portfolio"):
    data = fetch_stock_data(tickers, period="1y" if investment_horizon == "Short-term" else "5y")

    if model == "Modern Portfolio Theory (MPT)":
        weights, mean_returns, cov_matrix = optimize_portfolio(data, risk_tolerance)
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        st.subheader("📊 Optimized Portfolio (MPT)")
        portfolio_df = pd.DataFrame({"Stock": tickers, "Allocation (%)": weights * 100})
        st.dataframe(portfolio_df)
        st.write(f"**Expected Portfolio Return:** {portfolio_return:.2%}")
        st.write(f"**Expected Portfolio Risk:** {portfolio_risk:.2%}")

    elif model == "Momentum Strategy":
        momentum = (data.iloc[-1] / data.iloc[0]) - 1
        top_momentum_stocks = momentum.nlargest(5)
        st.subheader("🚀 Top Momentum Stocks")
        st.dataframe(top_momentum_stocks.to_frame(name="Momentum %"))
