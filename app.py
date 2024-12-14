import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
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
        # Prepare dataset summary for AI analysis
        dataset_summary = (
            "Dataset Overview:\n"
            f"Total Rows: {len(data)}\n"
            f"Total Columns: {len(data.columns)}\n"
            "Column Types:\n"
        )

        for col in data.columns:
            dataset_summary += f"{col}: {data[col].dtype}\n"

        # Sample of first few rows
        dataset_summary += "\nFirst few rows:\n"
        dataset_summary += data.head().to_string()

        # Query Gemini for dataset insights
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
    # Create a copy to avoid modifying original data
    processed_data = data.copy()

    # Encoders dictionary to store transformations
    encoders = {}

    for col in processed_data.columns:
        # Handle different data types intelligently
        if processed_data[col].dtype == 'object':
            # Check if column can be converted to numeric
            try:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='raise')
                continue
            except:
                # If not numeric, use label encoding
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str).fillna('Unknown'))
                encoders[col] = le

        # Handle missing values
        if processed_data[col].dtype in ['int64', 'float64']:
            processed_data[col].fillna(processed_data[col].median(), inplace=True)

    return processed_data, encoders


def select_best_model(X, y):
    """Dynamically select best model based on data characteristics"""
    # Determine problem type
    is_classification = len(np.unique(y)) <= 10  # Assume classification if few unique values

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


def create_interactive_prediction_plot(y_test, y_pred):
    """Create an interactive Plotly scatter plot of actual vs predicted values"""
    # Calculate prediction errors
    errors = np.abs(y_test - y_pred)

    # Create interactive scatter plot
    fig = go.Figure()

    # Scatter plot of actual vs predicted
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(
            size=10,
            color=errors,  # Color based on prediction error
            colorscale='RdYlBu',  # Red-Yellow-Blue color scale
            showscale=True,
            colorbar=dict(title='Absolute Error')
        ),
        text=[f'Actual: {actual:.2f}<br>Predicted: {pred:.2f}<br>Error: {err:.2f}'
              for actual, pred, err in zip(y_test, y_pred, errors)],
        hoverinfo='text'
    ))

    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='black', dash='dash')
    ))

    # Layout customization
    fig.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        hovermode='closest',
        template='plotly_white',
        height=600,
        width=800
    )

    # Add additional interactive elements
    st.plotly_chart(fig, use_container_width=True)

    # Error distribution histogram
    error_fig = px.histogram(
        x=errors,
        title='Prediction Error Distribution',
        labels={'x': 'Absolute Error'},
        marginal='box'  # Add box plot
    )
    st.plotly_chart(error_fig, use_container_width=True)


def process_and_predict(uploaded_file):
    try:
        # Load the dataset
        data = pd.read_csv(uploaded_file)

        # Initial data display
        st.subheader("🗂️ Dataset Overview")
        st.write(data.head())

        # AI-powered dataset analysis
        st.subheader("🤖 AI Dataset Insights")
        ai_insights = analyze_dataset_with_ai(data)
        st.write(ai_insights)

        # Preprocess the data
        st.subheader("🔄 Data Preprocessing")
        processed_data, encoders = preprocess_dataset(data)
        st.write("Processed Data:", processed_data.head())

        # Column selection for prediction
        st.subheader("📊 Select Columns")
        available_columns = list(processed_data.columns)
        target_column = st.selectbox("Select Target Column", available_columns)

        # Prepare features and target
        y = processed_data[target_column]
        X = processed_data.drop(columns=[target_column])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Select and train model
        model, scoring_func, report_func, is_classification = select_best_model(X_train, y_train)
        model.fit(X_train_scaled, y_train)

        # Predict and evaluate
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

        # Visualization
        st.subheader("📊 Prediction Visualization")
        create_interactive_prediction_plot(y_test, y_pred)

        # Gemini Q&A
        st.subheader("🤔 Ask About the Data")
        user_question = st.text_input("Ask Gemini a question about your dataset:")

        if user_question:
            # Prepare context for Gemini
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
