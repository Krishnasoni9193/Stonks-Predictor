import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import google.generativeai as genai
import os
from dotenv import load_dotenv
import requests
from datetime import datetime

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
        news_api_key = os.getenv("NEWS_API_KEY")
        if not news_api_key:
            st.error("NEWS_API_KEY is not set. Please configure your .env file.")
            return []

        query = f"{stock_symbol} stock market" if stock_symbol else "Indian stocks market"
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={news_api_key}"

        response = requests.get(url)
        if response.status_code == 200:
            news_data = response.json()
            news_articles = []
            for article in news_data.get('articles', [])[:10]:
                news_articles.append({
                    'title': article.get('title', 'No Title'),
                    'description': article.get('description', 'No description available'),
                    'url': article.get('url'),
                    'image': article.get('urlToImage', None),
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


def create_visualization(data, x_col, y_col, chart_type="scatter"):
    """
    Create dynamic visualizations based on user selection with flexible candlestick support
    """
    fig = None

    if chart_type == "scatter":
        fig = go.Figure(data=go.Scatter(x=data[x_col], y=data[y_col], mode='markers'))
        fig.update_layout(
            title=f"{y_col} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_dark"
        )

    elif chart_type == "candlestick":
        # Check if user has selected the date/time column for x-axis
        try:
            date_col = pd.to_datetime(data[x_col])
        except:
            st.error(f"Please select a valid date/time column for X-axis")
            return None

        # For candlestick, y_col will represent the price column
        # We'll let user select additional columns for high, low, and close
        st.info("For candlestick chart, please select the following columns:")

        # Create columns for selection
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            open_col = st.selectbox("Open Price Column", data.columns, index=data.columns.get_loc(y_col))
        with col2:
            high_col = st.selectbox("High Price Column", data.columns)
        with col3:
            low_col = st.selectbox("Low Price Column", data.columns)
        with col4:
            close_col = st.selectbox("Close Price Column", data.columns)

        # Create candlestick chart with selected columns
        fig = go.Figure(data=[go.Candlestick(
            x=date_col,
            open=data[open_col],
            high=data[high_col],
            low=data[low_col],
            close=data[close_col]
        )])

        fig.update_layout(
            title='Candlestick Chart',
            yaxis_title='Price',
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )

    elif chart_type == "line":
        fig = go.Figure(data=go.Scatter(x=data[x_col], y=data[y_col], mode='lines'))
        fig.update_layout(
            title=f"{y_col} Trend",
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_dark"
        )

    return fig


def add_technical_indicators(data, fig, x_column):
    """Add technical indicators to the chart"""
    # Get the currently selected OHLC columns from session state
    close_column = st.session_state.get('close_column', None)

    if not close_column:
        st.error("Please select Close Price Column first")
        return

    indicators = st.multiselect(
        "Select Technical Indicators",
        ["Moving Average", "RSI", "MACD"]
    )

    if "Moving Average" in indicators:
        ma_period = st.slider("Moving Average Period", 5, 50, 20)
        ma = data[close_column].rolling(window=ma_period).mean()
        fig.add_trace(go.Scatter(
            x=data[x_column],
            y=ma,
            name=f'MA{ma_period}',
            line=dict(color='orange', width=2)
        ))

    if "RSI" in indicators:
        rsi_period = st.slider("RSI Period", 5, 50, 14)
        delta = data[close_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        fig.add_trace(go.Scatter(
            x=data[x_column],
            y=rsi,
            name=f'RSI{rsi_period}',
            line=dict(color='purple', width=2)
        ))

    if "MACD" in indicators:
        exp1 = data[close_column].ewm(span=12, adjust=False).mean()
        exp2 = data[close_column].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        fig.add_trace(go.Scatter(
            x=data[x_column],
            y=macd,
            name='MACD',
            line=dict(color='cyan', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=data[x_column],
            y=signal,
            name='Signal',
            line=dict(color='yellow', width=2)
        ))


def create_visualization(data, x_col, y_col, chart_type="scatter"):
    """
    Create dynamic visualizations based on user selection
    """
    fig = None

    if chart_type == "scatter":
        fig = go.Figure(data=go.Scatter(x=data[x_col], y=data[y_col], mode='markers'))
        fig.update_layout(
            title=f"{y_col} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_dark"
        )

    elif chart_type == "candlestick":
        try:
            date_col = pd.to_datetime(data[x_col])
        except:
            st.error(f"Please select a valid date/time column for X-axis")
            return None

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            open_col = st.selectbox("Open Price Column", data.columns, index=data.columns.get_loc(y_col))
        with col2:
            high_col = st.selectbox("High Price Column", data.columns)
        with col3:
            low_col = st.selectbox("Low Price Column", data.columns)
        with col4:
            close_col = st.selectbox("Close Price Column", data.columns)

        # Store the selected columns in session state
        st.session_state['close_column'] = close_col

        fig = go.Figure(data=[go.Candlestick(
            x=date_col,
            open=data[open_col],
            high=data[high_col],
            low=data[low_col],
            close=data[close_col]
        )])

        fig.update_layout(
            title='Candlestick Chart',
            yaxis_title='Price',
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )

    elif chart_type == "line":
        fig = go.Figure(data=go.Scatter(x=data[x_col], y=data[y_col], mode='lines'))
        fig.update_layout(
            title=f"{y_col} Trend",
            xaxis_title=x_col,
            yaxis_title=y_col,
            template="plotly_dark"
        )

    return fig


def add_visualization_section(processed_data):
    """
    Add visualization controls and display charts
    """
    st.subheader("📊 Data Visualization")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["scatter", "candlestick", "line"]
        )

    with col2:
        x_column = st.selectbox(
            "Select X-axis Column",
            options=processed_data.columns,
            index=0
        )

    with col3:
        y_column = st.selectbox(
            "Select Y-axis Column",
            options=processed_data.columns,
            index=min(1, len(processed_data.columns) - 1)
        )

    fig = create_visualization(processed_data, x_column, y_column, chart_type)
    if fig:
        if chart_type == "candlestick":
            with st.expander("Technical Indicators"):
                add_technical_indicators(processed_data, fig, x_column)

        st.plotly_chart(fig, use_container_width=True)
# Update the create_visualization function to pass the close column
# def create_visualization(data, x_col, y_col, chart_type="scatter"):
#     """
#     Create dynamic visualizations based on user selection with flexible candlestick support
#     """
#     fig = None
#
#     if chart_type == "scatter":
#         fig = go.Figure(data=go.Scatter(x=data[x_col], y=data[y_col], mode='markers'))
#         fig.update_layout(
#             title=f"{y_col} vs {x_col}",
#             xaxis_title=x_col,
#             yaxis_title=y_col,
#             template="plotly_dark"
#         )
#
#     elif chart_type == "candlestick":
#         try:
#             date_col = pd.to_datetime(data[x_col])
#         except:
#             st.error(f"Please select a valid date/time column for X-axis")
#             return None
#
#         col1, col2, col3, col4 = st.columns(4)
#
#         with col1:
#             open_col = st.selectbox("Open Price Column", data.columns, index=data.columns.get_loc(y_col))
#         with col2:
#             high_col = st.selectbox("High Price Column", data.columns)
#         with col3:
#             low_col = st.selectbox("Low Price Column", data.columns)
#         with col4:
#             close_col = st.selectbox("Close Price Column", data.columns)
#
#         fig = go.Figure(data=[go.Candlestick(
#             x=date_col,
#             open=data[open_col],
#             high=data[high_col],
#             low=data[low_col],
#             close=data[close_col]
#         )])
#
#         fig.update_layout(
#             title='Candlestick Chart',
#             yaxis_title='Price',
#             template="plotly_dark",
#             xaxis_rangeslider_visible=False
#         )
#
#         # Store the close column for technical indicators
#         st.session_state['close_column'] = close_col
#
#     elif chart_type == "line":
#         fig = go.Figure(data=go.Scatter(x=data[x_col], y=data[y_col], mode='lines'))
#         fig.update_layout(
#             title=f"{y_col} Trend",
#             xaxis_title=x_col,
#             yaxis_title=y_col,
#             template="plotly_dark"
#         )
#
#     return fig
#
#
# def add_visualization_section(processed_data):
#     """
#     Add visualization controls and display charts
#     """
#     st.subheader("📊 Data Visualization")
#
#     col1, col2, col3 = st.columns([1, 1, 1])
#
#     with col1:
#         chart_type = st.selectbox(
#             "Select Chart Type",
#             ["scatter", "candlestick", "line"]
#         )
#
#     with col2:
#         x_column = st.selectbox(
#             "Select X-axis Column",
#             options=processed_data.columns,
#             index=0
#         )
#
#     with col3:
#         y_column = st.selectbox(
#             "Select Y-axis Column",
#             options=processed_data.columns,
#             index=min(1, len(processed_data.columns) - 1)
#         )
#
#     fig = create_visualization(processed_data, x_column, y_column, chart_type)
#     if fig:
#         if chart_type == "candlestick":
#             with st.expander("Technical Indicators"):
#                 # Use the stored close column for technical indicators
#                 close_column = st.session_state.get('close_column')
#                 if close_column:
#                     add_technical_indicators(processed_data, fig, x_column, close_column)
#                 else:
#                     st.error("Please select the Close Price Column first")
#
#         st.plotly_chart(fig, use_container_width=True)
def add_visualization_section(processed_data):
    """
    Add visualization controls and display charts
    """
    st.subheader("📊 Data Visualization")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["scatter", "candlestick", "line"]
        )

    with col2:
        x_column = st.selectbox(
            "Select X-axis Column",
            options=processed_data.columns,
            index=0
        )

    with col3:
        y_column = st.selectbox(
            "Select Y-axis Column",
            options=processed_data.columns,
            index=min(1, len(processed_data.columns) - 1)
        )

    fig = create_visualization(processed_data, x_column, y_column, chart_type)
    if fig:
        if chart_type == "candlestick":
            with st.expander("Technical Indicators"):
                add_technical_indicators(processed_data, fig, x_column)

        st.plotly_chart(fig, use_container_width=True)
def analyze_dataset_with_ai(data):
    """Use Gemini to analyze and understand the dataset."""
    try:
        # Get basic dataset info
        dataset_summary = (
            "Dataset Overview:\n"
            f"Total Rows: {len(data)}\n"
            f"Total Columns: {len(data.columns)}\n"
            "Column Types:\n"
        )

        # Add column information
        for col in data.columns:
            dataset_summary += f"{col}: {data[col].dtype}\n"
            # Add basic statistics for numerical columns
            if data[col].dtype in ['int64', 'float64']:
                dataset_summary += f"  - Mean: {data[col].mean():.2f}\n"
                dataset_summary += f"  - Min: {data[col].min()}\n"
                dataset_summary += f"  - Max: {data[col].max()}\n"

        # Limit the sample data to prevent large payloads
        sample_size = min(5, len(data))
        dataset_summary += f"\nSample of {sample_size} rows:\n"
        dataset_summary += data.head(sample_size).to_string(max_cols=10)

        # Truncate the summary if it's too long
        max_length = 40000  # Adjust this value based on API limits
        if len(dataset_summary) > max_length:
            dataset_summary = dataset_summary[:max_length] + "\n[Summary truncated due to length...]"

        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "You are an expert data analyst. Based on this limited dataset sample, provide insights:\n"
            "1. What are the key characteristics of this dataset?\n"
            "2. Suggest the most appropriate columns for prediction or classification.\n"
            "3. Identify any potential challenges in data preprocessing.\n\n"
            f"{dataset_summary}"
        )

        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"AI Analysis Error: {e}")
        return "Unable to generate AI insights. The dataset might be too large for analysis."


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
        # Read the file in chunks and check size
        file_details = uploaded_file.getvalue()
        file_size = len(file_details) / (1024 * 1024)  # Size in MB

        if file_size > 200:  # 200MB limit
            st.error(f"File size ({file_size:.1f}MB) exceeds 200MB limit. Please upload a smaller file.")
            return

        # Read data in chunks if it's large
        try:
            if file_size > 50:  # For files larger than 50MB
                data = pd.read_csv(uploaded_file, chunksize=10000)
                data = pd.concat([chunk for chunk in data])
            else:
                data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return

        # Sample data for large datasets
        if len(data) > 100000:  # If more than 100k rows
            st.warning("Large dataset detected. Using a sample for analysis.")
            data = data.sample(n=100000, random_state=42)

        st.subheader("🗂️ Dataset Overview")
        st.write("First few rows:")
        st.write(data.head())
        st.write(f"Total rows: {len(data):,}")
        st.write(f"Total columns: {len(data.columns)}")

        # Continue with rest of processing...

        # Add visualization section
        add_visualization_section(data)

        st.subheader("🤖 AI Dataset Insights")
        ai_insights = analyze_dataset_with_ai(data)
        st.write(ai_insights)

        st.subheader("🔄 Data Preprocessing")
        processed_data, encoders = preprocess_dataset(data)
        st.write("Processed Data:", processed_data.head())

        st.subheader("📊 Select Target Column")
        target_column = st.selectbox("Select Target Column", processed_data.columns)

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

        # Prediction vs Actual Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Predictions'
        ))
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Ideal',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title="Actual vs Predicted Values",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

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
# Streamlit App Layout
st.set_page_config(page_title="Intelligent Data Predictor with Stock News", layout="wide")

st.title("🧠 Intelligent Data Predictor")

col1, col2 = st.columns([3, 1])

with col1:
    st.write("Upload a CSV file for smart data analysis and prediction!")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        process_and_predict(uploaded_file)

with col2:
    st.subheader("📈 Real-Time Stock News")
    stock_symbol = st.text_input("Enter Stock Symbol (optional):")
    news_articles = fetch_indian_stock_news(stock_symbol)

    if news_articles:
        for article in news_articles[:5]:
            st.write(f"**{article['title']}**")
            if article['image']:
                st.image(article['image'], use_container_width=True)
            st.write(article['description'])
            st.write(f"Source: {article['source']} | {article['publishedAt']}")
            st.markdown(f"[Read more]({article['url']})")
            st.divider()
