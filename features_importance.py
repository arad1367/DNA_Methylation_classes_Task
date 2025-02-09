import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Read the CSV file
df = pd.read_csv('analysis_output_20250201_185244/important_features.csv')

# Create 3D visualization
fig = px.scatter_3d(
    df.head(1000),
    x=df.index,
    y='Score',
    z=df.groupby('Feature')['Score'].transform('rank'),
    color='Score',
    title='Feature Importance Distribution (Top 1000 Features)',
    labels={'x': 'Feature Index', 'y': 'Importance Score', 'z': 'Rank'},
    hover_data=['Feature']
)

# Update layout for better visualization
fig.update_layout(
    scene = dict(
        xaxis_title='Feature Index',
        yaxis_title='Importance Score',
        zaxis_title='Rank'
    ),
    width=1000,
    height=800,
    showlegend=True
)

# Save as HTML file
fig.write_html("feature_importance_3d.html")

# Display in browser
fig.show()