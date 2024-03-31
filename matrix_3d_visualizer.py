import streamlit as st
import pandas as pd
import plotly.graph_objects as go

#simple matrix visualize

# Function to parse user input into DataFrame
def parse_input(user_input):
    rows = user_input.split('\n')
    matrix = [list(map(int, row.split())) for row in rows]
    return pd.DataFrame(matrix)

# Streamlit application
st.title('Matrix Visualizer')

# User input for matrices A and B
matrix_a_input = st.text_area("Enter matrix A (rows separated by new lines, numbers separated by spaces):")
matrix_b_input = st.text_area("Enter matrix B (rows separated by new lines, numbers separated by spaces):")

# Parse user input into DataFrame
if matrix_a_input and matrix_b_input:
    matrix_a = parse_input(matrix_a_input)
    matrix_b = parse_input(matrix_b_input)

    # Button to trigger visualization
    if st.button('Visualize'):
        # Creates 3D plot
        fig = go.Figure(data=[go.Surface(z=matrix_a.values), go.Surface(z=matrix_b.values)])

        # Updates plot layout
        fig.update_layout(title='Matrix Visualization', autosize=False,
                          width=500, height=500,
                          margin=dict(l=65, r=50, b=65, t=90))

        # Displays plot
        st.plotly_chart(fig)

       
