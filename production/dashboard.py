import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Set the page configuration
st.set_page_config(page_title="Creative Dashboard", layout="wide")

# Function to generate sample data
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", periods=100)
    data = {
        "date": dates,
        "category": np.random.choice(["A", "B", "C"], size=100),
        "value1": np.random.randint(1, 100, size=100),
        "value2": np.random.randint(1, 100, size=100),
        "value3": np.random.randn(100).cumsum(),
    }
    return pd.DataFrame(data)

# Generate sample data
df = generate_data()

# Sidebar for user input
st.sidebar.header("User Input Parameters")
selected_category = st.sidebar.selectbox("Select Category", options=df["category"].unique())
selected_value = st.sidebar.selectbox("Select Value Type", options=["value1", "value2", "value3"])

# Filter data based on user input
filtered_df = df[df["category"] == selected_category]

# Layout: 3 cards at the top
data = {
    "value1": [10, 20, 30],
    "value2": [15, 25, 35],
    "value3": [12, 22, 32]
}
df = pd.DataFrame(data)

# Layout: 3 cards styled as "Resume", "Technical Interview", and "HR Interview"
data = {
    "value1": [10, 20, 30],
    "value2": [15, 25, 35],
    "value3": [12, 22, 32]
}
df = pd.DataFrame(data)

# Layout: 3 cards styled as "Resume", "Technical Interview", and "HR Interview"
st.title("Interview Metrics")
col1, col2, col3 = st.columns(3)

# Custom CSS for styling the cards
st.markdown("""
    <style>
        .metric-card {
            background-color: #b0b0b0;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
        }
        .metric-label {
            font-size: 20px;
            color: #333333;
        }
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Card 1: Resume
with col1:
    total_resume = df["value1"].sum()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{total_resume}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Resume</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Card 2: Technical Interview
with col2:
    total_technical = df["value2"].sum()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{total_technical}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Technical Interview</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Card 3: HR Interview
with col3:
    mean_hr = df["value3"].mean()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{mean_hr:.2f}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">HR Interview</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
# Define and display a circular progress bar
score = 75  # Sample score out of 100
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    title={'text': "Progress"},
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': "#4CAF50", 'thickness': 0.2},
           'steps': [
               {'range': [0, 50], 'color': "#FF6347"},
               {'range': [50, 100], 'color': "#90EE90"}]}))

# Custom CSS for centering and responsiveness
st.markdown("""
    <style>
        .plotly-container {
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
""", unsafe_allow_html=True)

# Display the Plotly chart
st.plotly_chart(fig, use_container_width=True)

# Layout: Graphs
st.subheader("Graphs")

# Line chart
fig1 = px.line(filtered_df, x="date", y=selected_value, title=f"Line Chart of {selected_value} for Category {selected_category}")
st.plotly_chart(fig1, use_container_width=True)

# Bar chart
fig2 = px.bar(filtered_df, x="date", y=selected_value, title=f"Bar Chart of {selected_value} for Category {selected_category}")
st.plotly_chart(fig2, use_container_width=True)

# Scatter plot
fig3 = px.scatter(filtered_df, x="value1", y="value2", color="category", title="Scatter Plot of Value1 vs Value2")
st.plotly_chart(fig3, use_container_width=True)

# Histogram
fig4 = px.histogram(df, x=selected_value, title=f"Histogram of {selected_value}")
st.plotly_chart(fig4, use_container_width=True)

# Adding some custom CSS for better visual appeal
st.markdown("""
    <style>
        .stMetric {
            border: 2px solid #4CAF50;
            border-radius: 5px;
            padding: 10px;
            margin: 5px;
            text-align: center;
            background-color: #f0f0f0;
        }
        .stPlotlyChart {
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)