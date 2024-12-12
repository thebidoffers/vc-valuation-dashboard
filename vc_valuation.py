# 1. All imports first
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# 2. Immediately after imports, set page config
st.set_page_config(page_title="VC Valuation Dashboard", layout="wide")

# 3. Function definitions
def calculate_valuation(time_horizon, terminal_revenue, exit_multiple, irr, 
                       future_rounds, dilution_per_round, investment_amount):
    """Calculate venture valuation using terminal value method"""
    terminal_value = terminal_revenue * exit_multiple
    cumulative_dilution = (1 - dilution_per_round) ** future_rounds
    discount_factor = (1 + irr) ** time_horizon
    pre_money_no_dilution = terminal_value / discount_factor
    pre_money_with_dilution = pre_money_no_dilution * cumulative_dilution
    post_money = pre_money_with_dilution + investment_amount
    
    return {
        'terminal_value': terminal_value,
        'pre_money_no_dilution': pre_money_no_dilution,
        'pre_money_with_dilution': pre_money_with_dilution,
        'post_money': post_money
    }

def create_waterfall_chart(series_data, series_name, terminal_revenue, exit_multiple, 
                          time_horizon, irr, dilution, future_rounds):
    """Create a waterfall chart showing valuation steps"""
    values = [
        series_data['terminal_value'],
        -(series_data['terminal_value'] - series_data['pre_money_no_dilution']),
        -(series_data['pre_money_no_dilution'] - series_data['pre_money_with_dilution'])
    ]
    
    fig = go.Figure(go.Waterfall(
        name="Valuation Steps",
        orientation="v",
        measure=["absolute", "relative", "relative"],
        x=["Terminal Value", "Time Value Effect", "Dilution Effect"],
        y=values,
        text=[f"${abs(v):,.0f}" for v in values],
        textposition="outside",
        decreasing={"marker": {"color": "red"}},
        increasing={"marker": {"color": "green"}},
        totals={"marker": {"color": "blue"}},
    ))
    fig.update_layout(
        title=f"{series_name} Valuation Waterfall",
        yaxis=dict(tickformat="$,.0f"),
        hovermode="x unified"
    )

    explanation = f"""
    **{series_name} Waterfall Chart**:

    1. **Terminal Value**: ${series_data['terminal_value']:,.0f}
       - Terminal Revenue: ${terminal_revenue:,.0f}
       - Exit Multiple: {exit_multiple}x

    2. **Time Value Effect**: -${(series_data['terminal_value'] - series_data['pre_money_no_dilution']):,.0f}
       - Discounted over {time_horizon} years at {irr*100:.1f}% IRR.

    3. **Dilution Effect**: -${(series_data['pre_money_no_dilution'] - series_data['pre_money_with_dilution']):,.0f}
       - Dilution per round: {dilution*100:.1f}% over {future_rounds} rounds.
    """
    return fig, explanation

def create_comparison_chart(series_a, series_b, series_c):
    """Create a bar chart comparing valuations"""
    fig = go.Figure(data=[
        go.Bar(name='Terminal Value', x=['Series A', 'Series B', 'Series C'],
               y=[series_a['terminal_value'], series_b['terminal_value'], series_c['terminal_value']]),
        go.Bar(name='Post-money', x=['Series A', 'Series B', 'Series C'],
               y=[series_a['post_money'], series_b['post_money'], series_c['post_money']])
    ])
    fig.update_layout(
        title='Valuation Comparison',
        barmode='group',
        yaxis=dict(tickformat="$,.0f"),
        height=600,
        legend=dict(orientation="h", y=1.02, x=1)
    )
    return fig

# Title
st.title("Venture Capital Valuation Dashboard")
st.write("Calculate startup valuations using the Terminal Value Method")

# Create three columns for inputs
col1, col2, col3 = st.columns(3)

# Function to format numbers with commas and parse back to float
def formatted_number_input(label, value, key):
    formatted_value = f"{value:,}"  # Display with commas
    input_value = st.text_input(label, formatted_value, key=key)
    return float(input_value.replace(",", ""))  # Parse back to raw number

# Series A Inputs
with col1:
    st.subheader("Series A")
    time_horizon_a = st.number_input("Time Horizon (Years)", min_value=1, value=4, key="th_a")
    terminal_revenue_a = formatted_number_input("Terminal Revenue", 100_000_000, "tr_a")
    exit_multiple_a = st.number_input("Exit Multiple", min_value=0.0, value=7.0, key="em_a")
    irr_a = st.number_input("Expected IRR", min_value=0.0, value=0.2, key="irr_a")
    future_rounds_a = st.number_input("Number of Future Rounds", min_value=0, value=4, key="fr_a")
    dilution_a = st.number_input("Estimated Dilution per Round", min_value=0.0, max_value=1.0, value=0.05, key="dil_a")
    investment_a = formatted_number_input("Investment Amount", 3_000_000, "inv_a")

# Similar inputs for Series B and Series C
with col2:
    st.subheader("Series B")
    time_horizon_b = st.number_input("Time Horizon (Years)", min_value=1, value=4, key="th_b")
    terminal_revenue_b = formatted_number_input("Terminal Revenue", 100_000_000, "tr_b")
    exit_multiple_b = st.number_input("Exit Multiple", min_value=0.0, value=7.0, key="em_b")
    irr_b = st.number_input("Expected IRR", min_value=0.0, value=0.2, key="irr_b")
    future_rounds_b = st.number_input("Number of Future Rounds", min_value=0, value=3, key="fr_b")
    dilution_b = st.number_input("Estimated Dilution per Round", min_value=0.0, max_value=1.0, value=0.05, key="dil_b")
    investment_b = formatted_number_input("Investment Amount", 3_000_000, "inv_b")

with col3:
    st.subheader("Series C")
    time_horizon_c = st.number_input("Time Horizon (Years)", min_value=1, value=3, key="th_c")
    terminal_revenue_c = formatted_number_input("Terminal Revenue", 100_000_000, "tr_c")
    exit_multiple_c = st.number_input("Exit Multiple", min_value=0.0, value=7.0, key="em_c")
    irr_c = st.number_input("Expected IRR", min_value=0.0, value=0.2, key="irr_c")
    future_rounds_c = st.number_input("Number of Future Rounds", min_value=0, value=2, key="fr_c")
    dilution_c = st.number_input("Estimated Dilution per Round", min_value=0.0, max_value=1.0, value=0.05, key="dil_c")
    investment_c = formatted_number_input("Investment Amount", 3_000_000, "inv_c")


# Perform calculations
series_a = calculate_valuation(time_horizon_a, terminal_revenue_a, exit_multiple_a, irr_a, future_rounds_a, dilution_a, investment_a)
series_b = calculate_valuation(time_horizon_b, terminal_revenue_b, exit_multiple_b, irr_b, future_rounds_b, dilution_b, investment_b)
series_c = calculate_valuation(time_horizon_c, terminal_revenue_c, exit_multiple_c, irr_c, future_rounds_c, dilution_c, investment_c)

# Display results
st.subheader("Valuation Results")
results = pd.DataFrame({
    'Metric': ['Terminal Value', 'Pre-money (No Dilution)', 'Pre-money (With Dilution)', 'Post-money'],
    'Series A': [series_a['terminal_value'], series_a['pre_money_no_dilution'], series_a['pre_money_with_dilution'], series_a['post_money']],
    'Series B': [series_b['terminal_value'], series_b['pre_money_no_dilution'], series_b['pre_money_with_dilution'], series_b['post_money']],
    'Series C': [series_c['terminal_value'], series_c['pre_money_no_dilution'], series_c['pre_money_with_dilution'], series_c['post_money']]
})

# Format results as currency
formatted_results = results.copy()  # Make a copy of results for formatting
for col in ['Series A', 'Series B', 'Series C']:
    formatted_results[col] = formatted_results[col].apply(lambda x: f"${x:,.2f}")

# Display the formatted table
st.table(formatted_results)


# Format results as currency
for col in ['Series A', 'Series B', 'Series C']:
    results[col] = results[col].apply(lambda x: f"${x:,.2f}")
st.table(results)

# Display Series A, B, and C charts
st.subheader("Waterfall Charts for Each Series")

# Create three columns for displaying charts
col1, col2, col3 = st.columns(3)

with col1:
    fig_a, exp_a = create_waterfall_chart(
        series_a, "Series A", terminal_revenue_a, exit_multiple_a, time_horizon_a, irr_a, dilution_a, future_rounds_a
    )
    st.plotly_chart(fig_a, use_container_width=True)
    st.info(exp_a)

with col2:
    fig_b, exp_b = create_waterfall_chart(
        series_b, "Series B", terminal_revenue_b, exit_multiple_b, time_horizon_b, irr_b, dilution_b, future_rounds_b
    )
    st.plotly_chart(fig_b, use_container_width=True)
    st.info(exp_b)

with col3:
    fig_c, exp_c = create_waterfall_chart(
        series_c, "Series C", terminal_revenue_c, exit_multiple_c, time_horizon_c, irr_c, dilution_c, future_rounds_c
    )
    st.plotly_chart(fig_c, use_container_width=True)
    st.info(exp_c)

# Comparison chart
st.subheader("Comparison of Valuations Across Series")
st.plotly_chart(create_comparison_chart(series_a, series_b, series_c), use_container_width=True)

