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

from service.ACPService import *

df= pd.read_csv("number-of-deaths-by-risk-factor.csv")

dfgps= pd.read_csv("world_country_gps.csv")

dfgps.rename(columns={'country': 'Entity'}, inplace=True)

model = SentenceTransformer('all-MiniLM-L6-v2') # Charger un modèle d'embedding pré-entraîné

# Créer des embeddings pour les noms de pays dans les deux dataframes
df_embeddings = model.encode(df['Entity'].tolist())
dfgps_embeddings = model.encode(dfgps['Entity'].tolist())


# Créer un dictionnaire pour stocker les correspondances
country_mapping = {}

# Pour chaque pays dans df, trouver le pays le plus similaire dans dfgps
for i, df_embedding in enumerate(df_embeddings):
    similarities = cosine_similarity([df_embedding], dfgps_embeddings)[0] # Calculer les similarités cosinus
    most_similar_index = np.argmax(similarities) # Trouver l'index du pays le plus similaire
    country_mapping[df['Entity'][i]] = dfgps['Entity'][most_similar_index] # Stocker la correspondance

# Créer une nouvelle colonne dans df pour stocker le nom de pays correspondant dans dfgps
df['Entity_matched'] = df['Entity'].map(country_mapping)

# Effectuer la jointure à gauche en utilisant la nouvelle colonne
merged_df = pd.merge(df, dfgps, left_on='Entity_matched', right_on='Entity', how='left')

# Supprimer la colonne temporaire 'Entity_matched'
merged_df.drop(columns=['Entity_matched'], inplace=True)

# Afficher les premières lignes du dataframe fusionné
#merged_df.head()

finalDF = merged_df[['Year','Air pollution','latitude','longitude']]
finalDF.head()


acp = ACPService(finalDF)
acp.biplot_with_adjusted_labels(
        score=acp.pca_res[:, 0:2],
        coeff=np.transpose(acp.components[0:2, :]),
        coeff_labels=df.columns,
        cat=acp.explained_variance[0:1],
        density=False
    )