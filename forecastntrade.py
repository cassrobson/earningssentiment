
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetExchange
import pandas as pd
import yfinance as yf
from pandas_datareader import data
import numpy as np
import requests
import clang as cl
from datetime import datetime, timedelta
from transformers import pipeline
from statistics import mean
from typing import Literal
import os
from pandas_market_calendars import get_calendar
import alpaca_trade_api as tradeapi
from earnings import get_customized_news
NYSE = get_calendar("NYSE")
def forecast(frame, correlation, earnings):
    alpaca_api_key = 'PKVVJTHP5OIYC73RAD2A'
    alpaca_secret_key = 'bEQYjHX50PvjOdHkQgFGeSaj8hYSMesazogKw4wY'
    base_url = 'https://paper-api.alpaca.markets'
    api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, base_url, api_version='v2')

    eod_api_key = '65e335de10b226.47129680'
    dates = []
    for i in frame.index:
        dates.append(i)

    nextdate = dates[3]
    estimate = frame.iat[3, 0]

    today = datetime.today()

    #todaystr = datetime.strftime(today, "%Y-%m-%d")
    
    tickerString = "NVDA"

    
    if today + timedelta(days=3) == nextdate:
            #pull news for last 5 days, aggregate sentiment, covered buy or sell based on correlation
        model_id = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        classify = pipeline('sentiment-analysis', model=model_id)
        scores = []
        for i in range(1,5):
                prerunningcount = 0
                date_obj = datetime.strptime(today, "%Y-%m-%d")
                pre_earnings_news_day = datetime.strftime(date_obj-timedelta(days=i+prerunningcount), "%Y-%m-%d")
                while not NYSE.valid_days(start_date=pre_earnings_news_day, end_date=pre_earnings_news_day).shape[0] == 1:
                    prerunningcount+=1
                    pre_earnings_news_day = datetime.strftime(date_obj-timedelta(days=i+prerunningcount), "%Y-%m-%d")
                else:
                    pre_earnings_news_day = datetime.strftime(date_obj-timedelta(days=i+prerunningcount), "%Y-%m-%d")
                pre_earnings_news = get_customized_news(tickerString, pre_earnings_news_day, pre_earnings_news_day, 30, api_key=eod_api_key)

                pre_earnings_news = [x for x in pre_earnings_news if "NVIDIA" or "NVDA" in x]
                if len(pre_earnings_news)>0:
                    scores_to_agg = []
                    for pre_story in pre_earnings_news:
                        pre_earnings_sentiment = classify(pre_story)
                        pre_earnings_sentiment_label = pre_earnings_sentiment[0].get('label')
                        
                        posmultiplier = 1
                        negmultiplier = -1
                        if pre_earnings_sentiment_label == "negative":
                            pre_earnings_sentiment_score = pre_earnings_sentiment[0].get('score') * negmultiplier
                        elif pre_earnings_sentiment_label == 'positive':
                            pre_earnings_sentiment_score = pre_earnings_sentiment[0].get('score') * posmultiplier
                        elif pre_earnings_sentiment_label == 'neutral':
                            pass
                        scores_to_agg.append(pre_earnings_sentiment_score)
                    abs_scores_to_agg = [abs(x) for x in scores_to_agg]
                    total_abs = sum(abs_scores_to_agg)
                    weights = [abs_score / total_abs for abs_score in abs_scores_to_agg]
                    weighted_sum = sum(score * weight for score, weight in zip(scores_to_agg, weights))
                    weighted_avg = weighted_sum / len (scores_to_agg)
                    agg_score = weighted_avg
                else:
                    agg_score = 0
                    data = pd.DataFrame({"Date":[pre_earnings_news_day], 'Sentiment_Score': [agg_score]})
                    biggest_frame_ever = pd.concat([biggest_frame_ever, data], ignore_index=True)
            
                scores.append(agg_score)
        total = 0
        count = 0
        for i in scores:
            count += 1
            total += i
            
        signal = total/count

        ticker = yf.Ticker('NVDA')
        info = ticker.get_info()
        
        for name, item in info.items():
            if name == 'currentPrice':
                current_price = item


        tobuy = round(current_price*0.9975, 2)
        take_profit_price = round(current_price*1.03)
        stop_loss_price = round(current_price*0.995)
        
        
        tosell = round(current_price*1.0025, 2) 

        if correlation > 0:
            if signal > 0:
                # Buy order with stop loss and take profit
                api.submit_order(
                    symbol='NVDA',  
                    qty=1,  
                    side='buy',  
                    type='limit',  
                    limit_price=tobuy,  
                    time_in_force='gtc',  
                    take_profit=dict(
                        limit_price=take_profit_price,
                    ),
                    stop_loss=dict(
                        stop_price=stop_loss_price,
                    )
                )

            elif signal < 0:
                # Place covered put with stop loss and take profit
                api.submit_order(
                    symbol='NVDA',  
                    qty=1,  
                    side='sell',
                    type='limit',  
                    limit_price=tosell, 
                    time_in_force='gtc',
                    order_class='simple',
                )
        else:
            print('NEGATIVE CORRELATION BETWEEN MEDIA SENTIMENT AND PRICE ACTION, THEREFORE MEDIA SENTIMENT IS NOT A GOOD MEASURE OF PRICE MOVEMENT')
