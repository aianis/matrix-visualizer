import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense

import plotly.graph_objects as go

# Function to parse user input into DataFrame
def parse_input(user_input):
    rows = user_input.split('\n')
    matrix = [list(map(int, row.split())) for row in rows]
    return pd.DataFrame(matrix)

# Streamlit application
st.title('Matrix Visualizer')

# Description of the app
st.write("""
The purpose of this app is to visualize the matrix transformation and matrices to better understand the concept of Linear Algebra. Just kidding! I just got bored of computing numbers cause I am not a calculator! 
Linear Algebra is not boring, it's cool actually :D
""")

# User input for matrices A and B
matrix_a_input = st.text_area("Enter matrix A (rows separated by new lines, numbers separated by spaces):")
matrix_b_input = st.text_area("Enter matrix B (rows separated by new lines, numbers separated by spaces):")

# Parse user input into DataFrame
if matrix_a_input and matrix_b_input:
    matrix_a = parse_input(matrix_a_input)
    matrix_b = parse_input(matrix_b_input)

    # Matrix addition and multiplication
    if matrix_a.shape == matrix_b.shape:
        matrix_sum = matrix_a + matrix_b
        matrix_product = matrix_a.dot(matrix_b)

        # Display matrix sum and product
        st.write('Matrix Sum:')
        st.write(matrix_sum)
        st.write('Matrix Product:')
        st.write(matrix_product)
    else:
        st.write('Matrix A and B must have the same dimensions for addition and multiplication.')

    # Check if matrices are square for determinant and eigenvalue calculations
    # if matrix_a.shape[0] == matrix_a.shape[1] and matrix_b.shape[0] == matrix_b.shape[1]:
    #     # Calculate determinants
    #     det_a = np.linalg.det(matrix_a)
    #     det_b = np.linalg.det(matrix_b)

    #     # Display determinants
    #     st.write(f'Determinant of matrix A: {det_a}')
    #     st.write(f'Determinant of matrix B: {det_b}')

    #     # Calculate eigenvalues and eigenvectors
    #     eigenvalues_a, eigenvectors_a = np.linalg.eig(matrix_a)
    #     eigenvalues_b, eigenvectors_b = np.linalg.eig(matrix_b)

    #     # Display eigenvalues and eigenvectors
    #     st.write(f'Eigenvalues of matrix A: {eigenvalues_a}, Eigenvectors: {eigenvectors_a}')
    #     st.write(f'Eigenvalues of matrix B: {eigenvalues_b}, Eigenvectors: {eigenvectors_b}')

       
    # Button to trigger visualization
    if st.button('Visualize'):
        # Creates 3D plot
        fig_3d = go.Figure(data=[go.Surface(z=matrix_a.values), go.Surface(z=matrix_b.values)])

        # Updates plot layout
        fig_3d.update_layout(title='3D Matrix Visualization', autosize=False,
                          width=500, height=500,
                          margin=dict(l=65, r=50, b=65, t=90))

        # Displays 3D plot
        st.plotly_chart(fig_3d)

        #################Matrix Transformation####################

         # Create a unit sphere
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        x = np.outer(np.sin(theta), np.cos(phi))
        y = np.outer(np.sin(theta), np.sin(phi))
        z = np.outer(np.cos(theta), np.ones_like(phi))

        # Apply the matrix transformation
        transformed_vectors = matrix_a.dot(np.array([x.flatten(), y.flatten(), z.flatten()]))

        # Convert transformed_vectors to a NumPy array
        transformed_vectors = transformed_vectors.to_numpy()

        # Create a 3D scatter plot of the transformed vectors
        fig_3d_transform = go.Figure(data=[go.Scatter3d(x=transformed_vectors[0, :], y=transformed_vectors[1, :], z=transformed_vectors[2, :], mode='markers')])
        fig_3d_transform.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), width=700, margin=dict(r=20, b=10, l=10, t=10))
        st.plotly_chart(fig_3d_transform)


        # Creates contour plot
        fig_contour = go.Figure(data=[go.Contour(z=matrix_a.values), go.Contour(z=matrix_b.values)])

        # Updates plot layout
        fig_contour.update_layout(title='Contour Matrix Visualization', autosize=False,
                          width=500, height=500,
                          margin=dict(l=65, r=50, b=65, t=90))

        # Displays contour plot
        st.plotly_chart(fig_contour)

        # Creates scatter plot
        fig_scatter = go.Figure(data=[go.Scatter(x=matrix_a.values.flatten(), y=matrix_b.values.flatten())])

        # Updates plot layout
        fig_scatter.update_layout(title='Scatter Matrix Visualization', autosize=False,
                          width=500, height=500,
                          margin=dict(l=65, r=50, b=65, t=90))

        # Displays scatter plot
        st.plotly_chart(fig_scatter)

