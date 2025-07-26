import requests
from bs4 import BeautifulSoup
import feedparser
from textblob import TextBlob
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import time
import re

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_stock_news(symbol, max_articles=10):
    """
    Fetch news articles for a given stock symbol from multiple sources
    
    Args:
        symbol (str): Stock symbol (e.g., 'RELIANCE')
        max_articles (int): Maximum number of articles to fetch
    
    Returns:
        list: List of news articles with title, content, date, source
    """
    articles = []
    
    try:
        # Google News RSS feed for the stock
        google_news_url = f"https://news.google.com/rss/search?q={symbol}+stock+india&hl=en-IN&gl=IN&ceid=IN:en"
        
        feed = feedparser.parse(google_news_url)
        
        for entry in feed.entries[:max_articles]:
            try:
                article = {
                    'title': entry.title,
                    'summary': entry.get('summary', ''),
                    'link': entry.link,
                    'published': entry.get('published', ''),
                    'source': entry.get('source', {}).get('title', 'Google News')
                }
                articles.append(article)
            except Exception as e:
                continue
        
        return articles
        
    except Exception as e:
        st.warning(f"Error fetching news for {symbol}: {str(e)}")
        return []

def analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob
    
    Args:
        text (str): Text to analyze
    
    Returns:
        dict: Sentiment analysis results
    """
    try:
        if not text or len(text.strip()) == 0:
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }
    
    except Exception as e:
        return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}

def get_news_sentiment_score(symbol):
    """
    Get overall news sentiment score for a stock
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        dict: News sentiment analysis results
    """
    try:
        articles = get_stock_news(symbol, max_articles=15)
        
        if not articles:
            return {
                'sentiment_score': 0,
                'sentiment_label': 'neutral',
                'confidence': 0,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            # Combine title and summary for analysis
            text = f"{article['title']} {article['summary']}"
            
            sentiment_result = analyze_sentiment(text)
            sentiments.append(sentiment_result)
            
            if sentiment_result['sentiment'] == 'positive':
                positive_count += 1
            elif sentiment_result['sentiment'] == 'negative':
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate overall sentiment score
        if sentiments:
            avg_polarity = sum(s['polarity'] for s in sentiments) / len(sentiments)
            avg_subjectivity = sum(s['subjectivity'] for s in sentiments) / len(sentiments)
            
            # Calculate confidence based on subjectivity and article count
            confidence = min(1.0, (1 - avg_subjectivity) * (len(articles) / 10))
            
            # Normalize sentiment score to 0-100 scale
            sentiment_score = (avg_polarity + 1) * 50  # Convert -1,1 to 0,100
            
            # Determine sentiment label
            if avg_polarity > 0.2:
                sentiment_label = 'bullish'
            elif avg_polarity < -0.2:
                sentiment_label = 'bearish'
            else:
                sentiment_label = 'neutral'
        else:
            sentiment_score = 50
            sentiment_label = 'neutral'
            confidence = 0
        
        return {
            'sentiment_score': round(sentiment_score, 2),
            'sentiment_label': sentiment_label,
            'confidence': round(confidence, 2),
            'article_count': len(articles),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'avg_polarity': round(avg_polarity, 3) if sentiments else 0,
            'recent_articles': articles[:5]  # Store recent articles for display
        }
    
    except Exception as e:
        st.error(f"Error analyzing news sentiment for {symbol}: {str(e)}")
        return {
            'sentiment_score': 50,
            'sentiment_label': 'neutral',
            'confidence': 0,
            'article_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0
        }

def get_market_sentiment():
    """
    Get overall market sentiment from major Indian stocks
    
    Returns:
        dict: Market sentiment analysis
    """
    try:
        major_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
        market_sentiments = []
        
        for stock in major_stocks:
            sentiment = get_news_sentiment_score(stock)
            if sentiment['article_count'] > 0:
                market_sentiments.append(sentiment['sentiment_score'])
        
        if market_sentiments:
            avg_market_sentiment = sum(market_sentiments) / len(market_sentiments)
            
            if avg_market_sentiment > 60:
                market_mood = 'bullish'
            elif avg_market_sentiment < 40:
                market_mood = 'bearish'
            else:
                market_mood = 'neutral'
            
            return {
                'market_sentiment_score': round(avg_market_sentiment, 2),
                'market_mood': market_mood,
                'stocks_analyzed': len(market_sentiments)
            }
        else:
            return {
                'market_sentiment_score': 50,
                'market_mood': 'neutral',
                'stocks_analyzed': 0
            }
    
    except Exception as e:
        return {
            'market_sentiment_score': 50,
            'market_mood': 'neutral',
            'stocks_analyzed': 0
        }

def clean_news_text(text):
    """Clean and preprocess news text for better sentiment analysis"""
    if not text:
        return ""
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    
    return text.lower()

def get_sector_sentiment(sector_stocks):
    """
    Get sentiment analysis for a specific sector
    
    Args:
        sector_stocks (list): List of stock symbols in the sector
    
    Returns:
        dict: Sector sentiment analysis
    """
    try:
        sector_sentiments = []
        
        for stock in sector_stocks[:5]:  # Analyze top 5 stocks in sector
            sentiment = get_news_sentiment_score(stock)
            if sentiment['article_count'] > 0:
                sector_sentiments.append(sentiment['sentiment_score'])
        
        if sector_sentiments:
            avg_sector_sentiment = sum(sector_sentiments) / len(sector_sentiments)
            
            if avg_sector_sentiment > 60:
                sector_mood = 'bullish'
            elif avg_sector_sentiment < 40:
                sector_mood = 'bearish'
            else:
                sector_mood = 'neutral'
            
            return {
                'sector_sentiment_score': round(avg_sector_sentiment, 2),
                'sector_mood': sector_mood,
                'stocks_analyzed': len(sector_sentiments)
            }
        else:
            return {
                'sector_sentiment_score': 50,
                'sector_mood': 'neutral',
                'stocks_analyzed': 0
            }
    
    except Exception as e:
        return {
            'sector_sentiment_score': 50,
            'sector_mood': 'neutral',
            'stocks_analyzed': 0
        }