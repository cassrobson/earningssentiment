import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot(df, earnings_dates):
    # Convert 'Date' column to datetime
    
    df['Date'] = pd.to_datetime(df['Date'])

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    # Plot candlestick chart
    fig.add_trace(go.Candlestick(x=df['Date'],
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='OHLC'),
                row=1, col=1)

    # Plot Sentiment Score
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Sentiment_Score']*100, mode='lines', name='Sentiment Score'),
                row=1, col=1)

    # Add Volume and % of 20D Adv subplot
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['% of 20D Adv'], mode='lines', name='% of 20D Adv'), row=2, col=1)

    # Update layout
    fig.update_layout(xaxis_rangeslider_visible=False)

    # Show the plot
    fig.show()