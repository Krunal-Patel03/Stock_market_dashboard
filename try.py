import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import pandas as pd
import streamlit as st

# Function to calculate color based on score
def get_color(score, max_score):
    if score >= max_score * 0.7:
        return "green"
    elif score >= max_score * 0.4:
        return "yellow"
    else:
        return "red"

# Function to display circular gauge
def display_circular_gauge(score, max_score):
    # Calculate progress percentage
    progress_percentage = (score / max_score) * 100

    # Create gauge figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge=dict(
            axis=dict(range=[None, max_score], tickvals=[0, max_score], ticks=''),
            bar=dict(color=get_color(score, max_score)),
            bgcolor="white",
            borderwidth=2,
            bordercolor="gray",
            steps=[
                dict(range=[0, max_score], color="white")
            ],
            threshold=dict(
                line=dict(color="red", width=2),
                thickness=0.75,
                value=score
            )
        ),
        number={'suffix': f'/{max_score}', 'font': {'size': 20}}
    ))
    fig.update_layout(width=250, height=250)
    # Display the gauge
    st.plotly_chart(fig)

def display_circular_gauge2(score, max_score):
    # Calculate progress percentage
    progress_percentage = (score / max_score) * 100

    # Create gauge figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge=dict(
            axis=dict(range=[None, max_score], tickvals=[0, max_score], ticks=''),
            bar=dict(color=get_color(score, max_score)),
            bgcolor="white",
            borderwidth=2,
            bordercolor="gray",
            steps=[
                dict(range=[0, max_score], color="white")
            ],
            threshold=dict(
                line=dict(color="red", width=2),
                thickness=0.75,
                value=score
            )
        ),
        number={'suffix': f'/{max_score}', 'font': {'size': 20}}
    ))
    fig.update_layout(width=250, height=250)


    # Return the fig object
    return fig

st.write(display_circular_gauge2(5.2,10))