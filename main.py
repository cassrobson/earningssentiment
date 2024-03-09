from earnings import *
from plot import *

from forecastntrade import forecast

def main():
    frame, earnings_dates = header()
    frame = frame.drop_duplicates()
    full_earnings = earnings_dates
    earnings_dates = earnings_dates.dropna()
    
    frame['day_move'] = frame['Open'] - frame['Close']
    correlation = frame['Sentiment_Score'].corr(frame['day_move'])
    print(correlation)
    frame.to_csv('output.csv')
    print(earnings_dates)
    plot(frame, earnings_dates)

    forecast(frame, correlation, full_earnings)
    


    

if __name__=="__main__":
    main()