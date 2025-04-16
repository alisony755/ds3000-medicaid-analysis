# How does the total Medicaid spending per drug compare across different companies?

# Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# Load the dataset
drug_df = pd.read_csv('drug_data.csv')

# Clustering features
clustering_features = [
    'medicaid_spending_2018',
    'medicaid_spending_2019',
    'medicaid_spending_2020',
    'medicaid_spending_2021',
    'medicaid_spending_2022'
]

# Preprocess data for clustering
df_cluster = drug_df.dropna(subset=clustering_features)
X_cluster = df_cluster[clustering_features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_cluster['pca1'] = X_pca[:, 0]
df_cluster['pca2'] = X_pca[:, 1]

# Create the scatter plot
fig = px.scatter(df_subset,
                 x='pca1',
                 y='pca2',
                 color='cluster_label',
                 hover_data='company',
                 title='Comparison of Medicaid Drug Spending Per Drug Across Companies (2018–2022)',
                 labels={
                     'pca1': 'Principal Component 1',
                     'pca2': 'Principal Component 2',
                     'cluster_label': 'Cluster Label'
                     },
                 color_discrete_sequence=['#478ce6', '#f74a7e', '#37ad82'])

# Set plot size
fig.update_layout(width=1300, height=600)

# Increase marker size and decrease opacity
fig.update_traces(marker=dict(size=15, opacity=0.7))

# Update font sizes
fig.update_layout(
    title_font=dict(size=24),
    legend_title_font=dict(size=20),
    legend_font=dict(size=16),
    xaxis_title_font=dict(size=20),
    yaxis_title_font=dict(size=20),
    xaxis_tickfont=dict(size=16),
    yaxis_tickfont=dict(size=16)
)

fig.show()
