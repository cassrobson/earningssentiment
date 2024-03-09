
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

trading_client = TradingClient('PKL49BV5A53T8JF5RV6D', 'YmN5cYyw0HKcMJwdm3r8JipBoJnrZvcidWIdKZiU')
NYSE = get_calendar("NYSE")
def header():
    eod_api_key = '65e335de10b226.47129680'
    nvda = yf.Ticker('NVDA')
    tickerString = 'NVDA'
    dates_frame = get_earnings_dates(nvda)
    print(dates_frame)
    hist = get_price_history(nvda)
    #sentiment_frame = market_sentiment_around_earnings(tickerString, dates_frame, eod_api_key)
    #sentiment_frame.to_pickle('sentiment_frame.pkl')
    #print(sentiment_frame)
    sentiment_frame = pd.read_pickle('sentiment_frame.pkl')
    frame_with_prices = add_prices(sentiment_frame, hist)
    return frame_with_prices, dates_frame
    

def get_account_info():
    # Get our account information.
    account = trading_client.get_account()

    # Check if our account is restricted from trading.
    if account.trading_blocked:
        print('Account is currently restricted from trading.')

    # Check how much money we can use to open new positions.
    print('${} is available as buying power.'.format(account.buying_power))

    balance_change = float(account.equity) - float(account.last_equity)
    print(f'Today\'s portfolio balance change: ${balance_change}')

def get_universe():
    lst = []
    universe = []
    godframe = pd.DataFrame()
    search = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
    assets = trading_client.get_all_assets(search)
    for item in assets:
        lst.append(dict(item))
    for dictionary in lst:
        if dictionary['exchange'] == AssetExchange.NYSE:
            universe.append(dictionary['symbol'])
    
    godframe.index = [x for x in universe]
    godframe['marketCap'] = np.nan
    memberIndex = 0
    print(godframe)
    """ for item in godframe.index:
        try:
            info = yf.Ticker(ticker=godframe.index[memberIndex]).info
            print(info)
            godframe['MCAP'] = info['state']
        except Exception as e:
            print("Error with ", godframe.index[memberIndex])
            print(e)
        memberIndex += 1 """
    

    """ info = yf.Ticker("AAPL").info
    print(info) """
    print(godframe)

def get_earnings_dates(nvda):
    dates = nvda.get_earnings_dates()
    return dates

def get_price_history(nvda):
    return nvda.history('2Y')

def add_prices(frame, hist):
    hist.index = [datetime.strftime(x, '%Y-%m-%d') for x in hist.index]
    hist['Date'] = hist.index
    #hist = hist.sort_index(ascending=False)
    hist = hist.drop(['Dividends', 'Stock Splits'], axis=1)
    hist["% of 20D Adv"] = hist['Volume'] / hist['Volume'].rolling(window=20).mean() * 100
    print(hist)
    print(frame)
    merged_df = pd.merge(frame, hist, on='Date', how='left')
    return merged_df



def market_sentiment_around_earnings(tickerString, dates, api_key):
    def convert_to_string(value):
        return str(value)
    
    dates.index = dates.index.map(convert_to_string)
    dates.index = [x[:10] for x in dates.index]
    dates = dates.sort_index(ascending=True)
    index = 0
    model_id = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    classify = pipeline('sentiment-analysis', model=model_id)
    biggest_frame_ever = pd.DataFrame()
   
    
    for earningsDate in dates.index:
        todaysDate = datetime.today()
        date_obj = datetime.strptime(earningsDate, "%Y-%m-%d")
        if date_obj < todaysDate:
            prerunningcount = 0
            for i in range(0, 30):
                date_obj = datetime.strptime(earningsDate, "%Y-%m-%d")
                pre_earnings_news_day = datetime.strftime(date_obj-timedelta(days=i+prerunningcount), "%Y-%m-%d")
                while not NYSE.valid_days(start_date=pre_earnings_news_day, end_date=pre_earnings_news_day).shape[0] == 1:
                    prerunningcount+=1
                    pre_earnings_news_day = datetime.strftime(date_obj-timedelta(days=i+prerunningcount), "%Y-%m-%d")
                else:
                    pre_earnings_news_day = datetime.strftime(date_obj-timedelta(days=i+prerunningcount), "%Y-%m-%d")
                pre_earnings_news = get_customized_news(tickerString, pre_earnings_news_day, pre_earnings_news_day, 30, api_key=api_key)
                

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
                    data = pd.DataFrame({"Date":[pre_earnings_news_day], 'Sentiment_Score': [agg_score]})
                    biggest_frame_ever = pd.concat([biggest_frame_ever, data], ignore_index=True)
                else:
                    agg_score = 0
                    data = pd.DataFrame({"Date":[pre_earnings_news_day], 'Sentiment_Score': [agg_score]})
                    biggest_frame_ever = pd.concat([biggest_frame_ever, data], ignore_index=True)
            postrunningcount = 0
            for i in range(0, 30): 
                date_obj = datetime.strptime(earningsDate, "%Y-%m-%d")
                post_earnings_news_day = datetime.strftime(date_obj-timedelta(days=i+postrunningcount), "%Y-%m-%d")
                while not NYSE.valid_days(start_date=post_earnings_news_day, end_date=post_earnings_news_day).shape[0] == 1:
                    postrunningcount+=1
                    post_earnings_news_day = datetime.strftime(date_obj-timedelta(days=i+postrunningcount), "%Y-%m-%d")
                else:
                    post_earnings_news_day = datetime.strftime(date_obj-timedelta(days=i+postrunningcount), "%Y-%m-%d")
                post_earnings_news = get_customized_news(tickerString, post_earnings_news_day, post_earnings_news_day, 30, api_key=api_key)
                post_earnings_news = [x for x in post_earnings_news if "NVIDIA" or "NVDA" in x]
                if len(post_earnings_news)>0:
                    scores_to_agg_post = []
                    for post_story in post_earnings_news:
                        post_earnings_sentiment = classify(post_story)
                        post_earnings_sentiment_label = post_earnings_sentiment[0].get('label')
                        
                        posmultiplier = 1
                        negmultiplier = -1
                        if post_earnings_sentiment_label == "negative":
                            post_earnings_sentiment_score = post_earnings_sentiment[0].get('score') * negmultiplier
                        elif post_earnings_sentiment_label == 'positive':
                            post_earnings_sentiment_score = post_earnings_sentiment[0].get('score') * posmultiplier
                        elif post_earnings_sentiment_label == 'neutral':
                            pass
                        scores_to_agg_post.append(post_earnings_sentiment_score)
                    abs_scores_to_agg_post = [abs(x) for x in scores_to_agg_post]
                    total_abs_post = sum(abs_scores_to_agg_post)
                    weights_post = [abs_score / total_abs_post for abs_score in abs_scores_to_agg_post]
                    weighted_sum_post = sum(score * weight for score, weight in zip(scores_to_agg_post, weights_post))
                    weighted_avg_post = weighted_sum_post / len (scores_to_agg_post)
                    agg_score_post = weighted_avg_post
                    data_post = pd.DataFrame({"Date":[post_earnings_news_day], 'Sentiment_Score': [agg_score_post]})
                    biggest_frame_ever = pd.concat([biggest_frame_ever, data_post], ignore_index=True)
                else:
                    agg_score_post = 0
                    data_post = pd.DataFrame({"Date":[post_earnings_news_day], 'Sentiment_Score': [agg_score_post]})
                    biggest_frame_ever = pd.concat([biggest_frame_ever, data_post], ignore_index=True)
        else:
            pass
        index += 1
    
    return biggest_frame_ever

def get_customized_news(stock, start_date, end_date, n_news, api_key, offset = 0):
    url = f'https://eodhistoricaldata.com/api/news?api_token={api_key}&s={stock}&limit={n_news}&offset={offset}&from={start_date}&to={end_date}'
    news_json = requests.get(url).json()
    
    news = []
    
    for i in range(len(news_json)):
        title = news_json[-i]['title']
        news.append(title)
        #print(cl('{}. '.format(i+1), attrs = ['bold']), '{}'.format(title))
    
    return news


