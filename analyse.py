import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from sklearn.linear_model import LinearRegression

from service.ACPService import *

# Fonction pour charger les fichiers CSV
def load_data():
    df = pd.read_csv("number-of-deaths-by-risk-factor.csv")
    dfgps = pd.read_csv("world_country_gps.csv")
    dfgps.rename(columns={'country': 'Entity'}, inplace=True)
    return df, dfgps

# Fonction pour créer les embeddings pour les pays
def create_embeddings(df, dfgps):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df_embeddings = model.encode(df['Entity'].tolist())
    dfgps_embeddings = model.encode(dfgps['Entity'].tolist())
    return df_embeddings, dfgps_embeddings

# Fonction pour créer la correspondance des pays
def create_country_mapping(df, dfgps, df_embeddings, dfgps_embeddings):
    country_mapping = {}
    for i, df_embedding in enumerate(df_embeddings):
        similarities = cosine_similarity([df_embedding], dfgps_embeddings)[0]
        most_similar_index = np.argmax(similarities)
        country_mapping[df['Entity'][i]] = dfgps['Entity'][most_similar_index]
    return country_mapping

# Fonction pour fusionner les DataFrames
def merge_data(df, dfgps, country_mapping):
    df['Entity_matched'] = df['Entity'].map(country_mapping)
    merged_df = pd.merge(df, dfgps, left_on='Entity_matched', right_on='Entity', how='left')
    merged_df.drop(columns=['Entity_matched'], inplace=True)
    return merged_df

# Fonction pour calculer l'inertie pour différents nombres de clusters et tracer le graphe du coude
def elbow_method(df, columns, max_clusters):
    inertias = []  # Liste pour stocker les inerties pour chaque k
    K = range(1, max_clusters + 1)  # Nombre de clusters à tester
    
    # Calcul de l'inertie pour chaque valeur de k
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df[columns])
        inertias.append(kmeans.inertia_)
    
    # Tracé du graphe du coude
    plt.figure(figsize=(8, 6))
    plt.plot(K, inertias, marker='o', linestyle='--')
    plt.xlabel('Nombre de clusters')
    plt.ylabel("Inertie intra-cluster (WCSS)")
    plt.title("Méthode du coude pour déterminer le nombre optimal de clusters")
    plt.show()

    return inertias  # Retourne les inerties pour chaque k

def plot_clusters_on_map(df, n_clusters=7, feature_columns=['latitude', 'longitude']):
    # Appliquer le clustering KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df.loc[:, 'Cluster'] = kmeans.fit_predict(df[feature_columns])

    # Conversion en GeoDataFrame
    geo_clusters = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['longitude'], df['latitude'])
    )

    # Tracer la carte
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)

    # Tracé des clusters sur la carte
    scatter = ax.scatter(
        df['longitude'], df['latitude'],
        c=df['Cluster'], cmap='tab10', s=50, transform=ccrs.PlateCarree()
    )

    # Titre et légende
    ax.set_title("Clusters de pollution de l'air sur la carte du monde", fontsize=16)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.show()

    return geo_clusters  # Retourne le GeoDataFrame contenant les clusters

def apply_kmeans_clustering(df, n_clusters=7, feature_columns=['latitude', 'longitude']):
    # Vérifier si df est une vue et faire une copie si nécessaire
    if df._is_copy:
        df = df.copy()
    
    # Vérifier si la colonne 'Cluster' existe déjà et la signaler
    if 'Cluster' in df.columns:
        print("La colonne 'Cluster' existe déjà. Elle sera remplacée.")
    
    # Initialisation du modèle KMeans avec n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Appliquer le clustering sur les données spécifiées
    df.loc[:, 'Cluster'] = kmeans.fit_predict(df[feature_columns])
    
    # Retourner le DataFrame avec les clusters ajoutés
    return df

def linear_regression_by_cluster(df, cluster_column='Cluster', feature_column='Year', target_column='Air pollution'):
    resultats = {}

    # Régression linéaire par cluster
    for cluster in df[cluster_column].unique():
        cluster_data = df[df[cluster_column] == cluster]

        X = cluster_data[[feature_column]]
        y = cluster_data[target_column]

        model = LinearRegression()
        model.fit(X, y)

        # Stockage des coefficients du modèle pour chaque cluster
        resultats[cluster] = {
            'intercept': model.intercept_,
            'coef': model.coef_[0]
        }

    return resultats

def predict(year, latitude, longitude, kmeans, resultats):
    # Prédire le cluster pour les coordonnées spécifiées
    cluster = kmeans.predict([[latitude, longitude]])[0]
    
    # Extraire les résultats du modèle de régression pour ce cluster
    model_data = resultats[cluster]
    
    # Calculer la prédiction (modèle linéaire : intercept + coef * year)
    prediction = model_data['intercept'] + model_data['coef'] * year
    
    return prediction

# Fonction pour tracer les régressions et les prédictions
def plot_regression_clusters(finalDF, n_clusters=7, feature_columns=['latitude', 'longitude'], end_year=2050):
    # Appliquer le clustering KMeans sur les colonnes spécifiées
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    finalDF['Cluster'] = kmeans.fit_predict(finalDF[feature_columns])

    # Régression linéaire par cluster
    resultats = {}

    for cluster in finalDF['Cluster'].unique():
        cluster_data = finalDF[finalDF['Cluster'] == cluster]

        X = cluster_data[['Year']]
        y = cluster_data['Air pollution']

        model = LinearRegression()
        model.fit(X, y)

        # Stocker les coefficients du modèle
        resultats[cluster] = {
            'intercept': model.intercept_,
            'coef': model.coef_[0]
        }

    # Tracer les régressions par cluster
    clusters = finalDF['Cluster'].unique()
    plt.figure(figsize=(12, 8))

    for cluster in clusters:
        
        # Données du cluster
        cluster_data = finalDF[finalDF['Cluster'] == cluster]
        X = cluster_data[['Year']].values
        y = cluster_data['Air pollution'].values

        # Prédictions basées sur le modèle
        model_data = resultats[cluster]
        
        # Générer la plage des années pour la régression
        X_range = np.linspace(X.min(), end_year, 100).reshape(-1, 1)  # jusqu'à l'année spécifiée
        y_pred = model_data['intercept'] + model_data['coef'] * X_range

        # Tracé des points de régression
        plt.scatter(X, y, label=f'Cluster {cluster} (données)', alpha=0.6)
        plt.plot(X_range, y_pred, label=f'Regression Cluster {cluster}', linestyle='--')

        # Prédictions pour les années futures
        future_predictions = [predict(year, cluster_data['latitude'].iloc[0], cluster_data['longitude'].iloc[0], kmeans, resultats) for year in range(X.max(), end_year + 1)]
        plt.scatter(range(X.max(), end_year + 1), future_predictions, color='red', marker='x', label=f'Prédictions futures Cluster {cluster}')

    # Configuration du graphique
    plt.title("Régressions par Cluster et Prédictions futures")
    plt.xlabel("Année")
    plt.ylabel("Nombre de morts par pollution de l'air")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Main
def main():
    # Charger les données
    df, dfgps = load_data()

    # Créer les embeddings
    df_embeddings, dfgps_embeddings = create_embeddings(df, dfgps)

    # Créer la correspondance des pays
    country_mapping = create_country_mapping(df, dfgps, df_embeddings, dfgps_embeddings)

    # Fusionner les DataFrames
    merged_df = merge_data(df, dfgps, country_mapping)

    # Extraire la partie nécessaire des données
    finalDF = merged_df[['Year', 'Air pollution', 'latitude', 'longitude']]
    print(finalDF.head())

    # Effectuer l'ACP et le biplot
    acp = ACPService(finalDF)
    acp.biplot_with_adjusted_labels(
        score=acp.pca_res[:, 0:2],
        coeff=np.transpose(acp.components[0:2, :]),
        coeff_labels=df.columns,
        cat=acp.explained_variance[0:1],
        density=False
    )

    # Appliquer la méthode du coude pour déterminer le nombre optimal de clusters
    elbow_method(finalDF, columns=['latitude', 'longitude'], max_clusters=10)

    # Tracer les clusters sur une carte
    geo_clusters = plot_clusters_on_map(finalDF, n_clusters=7)

    # Appliquer le clustering et obtenir les clusters
    finalDF_with_clusters = apply_kmeans_clustering(finalDF, n_clusters=7)
    
    # Afficher le DataFrame avec la colonne 'Cluster' ajoutée
    print(finalDF_with_clusters.head())

    # Effectuer la régression linéaire par cluster
    resultats = linear_regression_by_cluster(finalDF_with_clusters)

    # Initialiser et ajuster le modèle KMeans
    kmeans = KMeans(n_clusters=7, random_state=42)
    kmeans.fit(finalDF[['latitude', 'longitude']])

    # Exemple de prédiction
    latitude = 48.8566  # Latitude de Paris
    longitude = 2.3522  # Longitude de Paris
    year = 2030  # Année de la prédiction
    
    prediction = predict(year, latitude, longitude, kmeans, resultats)
    print(prediction)

    # Appliquer la fonction de régression et de clustering
    plot_regression_clusters(finalDF, n_clusters=7, end_year=2050)

if __name__ == "__main__":
    main()
