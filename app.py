import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import google.generativeai as genai
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate the API key
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set. Please configure your .env file.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Function to fetch real-time stock news
def fetch_indian_stock_news(stock_symbol=None):
    """
    Fetch live news for Indian stocks using NewsAPI and include images.
    """
    try:
        # Get NewsAPI key from environment variables
        news_api_key = os.getenv("NEWS_API_KEY")

        if not news_api_key:
            st.error("NEWS_API_KEY is not set. Please configure your .env file.")
            return []

        # Prepare query
        query = f"{stock_symbol} stock market" if stock_symbol else "Indian stocks market"

        # Construct API URL
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={news_api_key}"

        # Make API request
        response = requests.get(url)

        if response.status_code == 200:
            news_data = response.json()

            # Prepare news articles
            news_articles = []
            for article in news_data.get('articles', [])[:10]:  # Limit to 10 articles
                news_articles.append({
                    'title': article.get('title', 'No Title'),
                    'description': article.get('description', 'No description available'),
                    'url': article.get('url'),
                    'image': article.get('urlToImage', None),  # Include image URL
                    'publishedAt': article.get('publishedAt'),
                    'source': article.get('source', {}).get('name', 'Unknown')
                })

            return news_articles
        else:
            st.error(f"Error fetching news: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"News Fetch Error: {e}")
        return []

def analyze_dataset_with_ai(data):
    """Use Gemini to analyze and understand the dataset."""
    try:
        dataset_summary = (
            "Dataset Overview:\n"
            f"Total Rows: {len(data)}\n"
            f"Total Columns: {len(data.columns)}\n"
            "Column Types:\n"
        )

        for col in data.columns:
            dataset_summary += f"{col}: {data[col].dtype}\n"

        dataset_summary += "\nFirst few rows:\n"
        dataset_summary += data.head().to_string()

        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "You are an expert data analyst. Analyze this dataset and provide insights:\n"
            "1. What are the key characteristics of this dataset?\n"
            "2. Suggest the most appropriate columns for prediction or classification.\n"
            "3. Identify any potential challenges in data preprocessing.\n\n"
            f"{dataset_summary}"
        )

        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"AI Analysis Error: {e}")
        return "Unable to generate AI insights."


def preprocess_dataset(data):
    """Intelligent dataset preprocessing"""
    processed_data = data.copy()
    encoders = {}

    for col in processed_data.columns:
        if processed_data[col].dtype == 'object':
            try:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='raise')
                continue
            except:
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str).fillna('Unknown'))
                encoders[col] = le

        if processed_data[col].dtype in ['int64', 'float64']:
            processed_data[col].fillna(processed_data[col].median(), inplace=True)

    return processed_data, encoders


def select_best_model(X, y):
    """Dynamically select best model based on data characteristics"""
    is_classification = len(np.unique(y)) <= 10

    if is_classification:
        st.write("🔍 Detected Classification Problem")
        model = LogisticRegression(max_iter=1000)
        scoring_func = accuracy_score
        report_func = classification_report
    else:
        st.write("🔍 Detected Regression Problem")
        model = RandomForestRegressor()
        scoring_func = r2_score
        report_func = mean_squared_error

    return model, scoring_func, report_func, is_classification


def process_and_predict(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("🗂️ Dataset Overview")
        st.write(data.head())

        st.subheader("🤖 AI Dataset Insights")
        ai_insights = analyze_dataset_with_ai(data)
        st.write(ai_insights)

        st.subheader("🔄 Data Preprocessing")
        processed_data, encoders = preprocess_dataset(data)
        st.write("Processed Data:", processed_data.head())

        st.subheader("📊 Select Columns")
        available_columns = list(processed_data.columns)
        target_column = st.selectbox("Select Target Column", available_columns)

        y = processed_data[target_column]
        X = processed_data.drop(columns=[target_column])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model, scoring_func, report_func, is_classification = select_best_model(X_train, y_train)
        model.fit(X_train_scaled, y_train)

        st.subheader("🎯 Prediction Results")
        y_pred = model.predict(X_test_scaled)

        if is_classification:
            score = scoring_func(y_test, y_pred)
            st.write(f"Accuracy: {score:.2%}")
            st.write("Classification Report:")
            st.text(report_func(y_test, y_pred))
        else:
            score = scoring_func(y_test, y_pred)
            st.write(f"R² Score: {score:.2%}")
            st.write(f"Mean Squared Error: {report_func(y_test, y_pred)}")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title("Actual vs Predicted Values")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        st.pyplot(plt)

        st.subheader("🤔 Ask About the Data")
        user_question = st.text_input("Ask Gemini a question about your dataset:")

        if user_question:
            context = (
                f"Dataset Overview:\n"
                f"Total Rows: {len(data)}\n"
                f"Total Columns: {len(data.columns)}\n"
                f"Target Column: {target_column}\n"
                f"Prediction Type: {'Classification' if is_classification else 'Regression'}\n"
                f"Model Performance: {score:.2%}"
            )

            try:
                model_gemini = genai.GenerativeModel('gemini-1.5-flash')
                response = model_gemini.generate_content(f"{context}\n\nUser's question: {user_question}")
                st.write("Gemini's Response:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Gemini API Error: {e}")

    except Exception as e:
        st.error(f"Processing Error: {e}")


# Streamlit App Layout
st.set_page_config(page_title="Intelligent Data Predictor with Stock News", layout="wide")

st.title("🧠 Stonks Predictor Intelligence")

col1, col2 = st.columns([3, 1])

with col1:
    st.write("Upload a file with clean and beautiful attributes,and that's it! Give me some seconds to analyze it for you!")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        process_and_predict(uploaded_file)

with col2:
    st.subheader("📈Live Stock News")
    news_articles = fetch_indian_stock_news()

    if news_articles:
        for article in news_articles[:5]:  # Display top 5 articles
            st.write(f"**{article['title']}**")
            st.write(article['description'])
            st.markdown(f"[Read more]({article['url']})")
