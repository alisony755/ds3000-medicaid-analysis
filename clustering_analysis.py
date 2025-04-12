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
clustering_features = ['medicaid_spending_2018', 'medicaid_spending_2019', 
                       'medicaid_spending_2020', 'medicaid_spending_2021', 
                       'medicaid_spending_2022']

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
fig = px.scatter(df_cluster,
                 x='pca1', 
                 y='pca2', 
                 color='kmeans_cluster', 
                 hover_data=['company'],
                 title="Clustering of Pharmaceutical Companies based on Medicaid Spending",
                 color_continuous_scale='Viridis')

fig.show()
