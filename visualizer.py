import numpy as np
import plotly.graph_objects as go
import glob
 

def load_and_plot_npy_files():

    # Get all .npy files in the folder
    npy_files = sorted(glob.glob("npyfiles/*.npy"))

    all_points = [np.load(f, allow_pickle=True) for f in npy_files]

    for points in all_points:
        # Load your (3618, 3) NumPy array
        
        print(points.shape)
        points = points.reshape(6030,3)
        # Extract X, Y, and Z coordinates
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Create a 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.8)
        )])
        
        # Update layout
        fig.update_layout(
            title="3D Scatter Plot",
            scene=dict(
                xaxis_title="X-axis",
                yaxis_title="Y-axis",
                zaxis_title="Z-axis"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        # Show the plot
        fig.show()