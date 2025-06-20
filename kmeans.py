import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.metrics import silhouette_score
import numpy as np

# Fichiers à utiliser
files = [
    'data_a_utiliser/leg_2017.csv',
    'data_a_utiliser/leg_2022.csv',
    'data_a_utiliser/leg_2024.csv'
]

# Chargement et concaténation des données
frames = []
for f in files:
    df = pd.read_csv(f)
    # Ajoute l'année à chaque DataFrame
    year = f.split('_')[-1].split('.')[0]
    df['ANNEE'] = year
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# Colonnes d'identification
id_cols = ['Code du département', 'Code de la circonscription', 'ANNEE']

# Colonnes features (toutes sauf id)
feature_cols = [col for col in data.columns if col not in id_cols]

# On ne garde que les features numériques
X = data[feature_cols].fillna(0)

# On va pivoter les données pour avoir un vecteur par circo sur 3 élections
# On garde les colonnes d'identification sans l'année
id_circo = ['Code du département', 'Code de la circonscription']

# On pivote pour avoir une colonne par score et par année
pivot = data.pivot_table(
    index=id_circo,
    columns='ANNEE',
    values=[col for col in data.columns if col not in id_cols],
    fill_value=0
)

# On a maintenant un multi-index de colonnes (score, année), on aplatit
pivot.columns = [f"{score}_{annee}" for score, annee in pivot.columns]

# On réinitialise l'index pour avoir un DataFrame classique
pivot = pivot.reset_index()

# On retire les colonnes d'identification pour le clustering
X = pivot.drop(columns=id_circo)

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

pivot['cluster'] = clusters


# Visualisation des clusters avec réduction de dimension (PCA)

# Réduction à 2 dimensions pour affichage
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pivot['PCA1'] = X_pca[:, 0]
pivot['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pivot,
    x='PCA1', y='PCA2',
    hue='cluster',
    palette='tab10',
    legend='full',
    s=60
)
plt.title('Clustering des circonscriptions')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.legend(title='Cluster')
plt.tight_layout()
#plt.show()

# Visualisation interactive avec affichage du nom de la circo et du département au survol
pivot['circo_label'] = pivot['Code du département'].astype(str) + '-' + pivot['Code de la circonscription'].astype(str)

fig = px.scatter(
    pivot,
    x='PCA1',
    y='PCA2',
    color='cluster',
    hover_name='circo_label',
    hover_data={
        'Code du département': True,
        'Code de la circonscription': True,
        'cluster': True
    },
    title='Clustering des circonscriptions législatives françaises avec KMeans',
    labels={'PCA1': 'Composante principale 1', 'PCA2': 'Composante principale 2'}
)
# Augmenter la taille et l'opacité des points pour faciliter le survol
fig.update_traces(marker=dict(size=18, opacity=0.95, line=dict(width=1, color='black')))

# Supprimer tous les appels à fig.show() pour éviter l'ouverture d'un onglet Plotly séparé
# (Gardez uniquement la visualisation dans Dash via dcc.Graph)

# Ajout des infos département et région pour le hover Plotly
# Chargement du fichier de correspondance
regions = pd.read_csv('data_a_utiliser/departments_regions_france_2016.csv', dtype=str)
regions['departmentCode'] = regions['departmentCode'].str.lower()
regions['departmentName'] = regions['departmentName'].str.strip()
regions['regionName'] = regions['regionName'].str.strip()

# Harmonisation des codes département dans pivot (zéros à gauche, minuscules pour corse)
def format_dep(dep):
    dep = str(dep).lower()
    if dep.isdigit():
        return dep.zfill(2)
    return dep
pivot['dep_code_str'] = pivot['Code du département'].apply(format_dep)

# Jointure pour récupérer le nom du département et la région
pivot = pivot.merge(regions, left_on='dep_code_str', right_on='departmentCode', how='left')

# Visualisation interactive enrichie et esthétique (français, sans contours, avec grille)
pivot['circo_label'] = pivot['Code du département'].astype(str) + '-' + pivot['Code de la circonscription'].astype(str)

fig = px.scatter(
    pivot,
    x='PCA1',
    y='PCA2',
    color='cluster',
    hover_name='circo_label',
    hover_data={
        'Code du département': True,
        'departmentName': True,
        'regionName': True,
        'Code de la circonscription': True,
        'cluster': True,
        'PCA1': False,
        'PCA2': False
    },
    title='<b>Clustering des circonscriptions législatives françaises (KMeans, PCA 2D)</b>',
    labels={
        'PCA1': '',
        'PCA2': '',
        'departmentName': 'Département',
        'regionName': 'Région',
        'Code du département': 'Code département',
        'Code de la circonscription': 'Circonscription',
        'cluster': 'Cluster'
    },
    template='plotly_white',
    opacity=0.95
)
fig.update_traces(
    marker=dict(size=12, opacity=0.95, line=dict(width=1, color='black')),  # points plus petits, léger contour noir
    selector=dict(mode='markers')
)
fig.update_layout(
    font=dict(family='Arial', size=15),
    title_font=dict(size=22, family='Arial Black'),
    legend=dict(title='Cluster', font=dict(size=14)),
    xaxis=dict(showgrid=True, zeroline=False, showticklabels=False, gridcolor='#e5e5e5'),
    yaxis=dict(showgrid=True, zeroline=False, showticklabels=False, gridcolor='#e5e5e5'),
    plot_bgcolor='#FFF8E7',  # beige très clair
    paper_bgcolor='#FFF8E7'  # beige très clair
)
# Remplacement des = par : dans les tooltips
fig.for_each_trace(lambda t: t.update(hovertemplate=t.hovertemplate.replace('=', ':')))

# Affichage de l'inertie (somme des distances intra-cluster)
print(f"Inertie du KMeans : {kmeans.inertia_:.2f}")

# Score de silhouette
sil_score = silhouette_score(X_scaled, clusters)
print(f"Score de silhouette : {sil_score:.3f} (proche de 1 = bons clusters, proche de 0 = recouvrement)")

# Comparaison à un clustering aléatoire
random_labels = np.random.randint(0, n_clusters, size=len(X_scaled))
sil_score_random = silhouette_score(X_scaled, random_labels)
print(f"Score de silhouette (clustering aléatoire) : {sil_score_random:.3f}")

def trouver_circos_proches(num_departement, num_circo, n=3):
    # Formatage du code département
    def format_dep(dep):
        dep = str(dep).lower()
        if dep.isdigit():
            return dep.zfill(2)
        return dep
    dep_code = format_dep(num_departement)
    circo_code = str(num_circo)
    # Trouver l'index de la circo demandée
    mask = (pivot['dep_code_str'] == dep_code) & (pivot['Code de la circonscription'].astype(str) == circo_code)
    if not mask.any():
        print("Circonscription non trouvée.")
        return
    idx = pivot[mask].index[0]
    # Calcul des distances euclidiennes dans l'espace PCA
    dists = np.linalg.norm(pivot[['PCA1', 'PCA2']].values - pivot.loc[idx, ['PCA1', 'PCA2']].values, axis=1)
    # Exclure la circo elle-même
    dists[idx] = np.inf
    proches_idx = np.argsort(dists)[:n]
    print(f"Les {n} circonscriptions les plus proches de {num_departement}-{num_circo} sont :")
    for i in proches_idx:
        row = pivot.iloc[i]
        print(f"  - {row['Code du département']}-{row['Code de la circonscription']} ({row['departmentName']}, {row['regionName']}) | Distance : {dists[i]:.2f}")

# Exemple d'utilisation :
# trouver_circos_proches('75', 1)

# Ajout d'une interface Dash pour une recherche interactive des circonscriptions les plus proches
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Liste des circonscriptions pour la barre de recherche
pivot['circo_search'] = pivot['Code du département'].astype(str) + '-' + pivot['Code de la circonscription'].astype(str) + \
    ' (' + pivot['departmentName'] + ', ' + pivot['regionName'] + ')'

app.layout = dbc.Container([
    html.H2("Recherche de circonscriptions proches", style={"marginTop": 30}),
    dcc.Dropdown(
        id='circo-dropdown',
        options=[{'label': str(c), 'value': str(i)} for i, c in enumerate(pivot['circo_search'])],
        placeholder="Choisissez une circonscription...",
        style={"width": "100%", "marginBottom": 20}
    ),
    html.Div(id='proches-output'),
    dcc.Graph(
        id='scatter-graph',
        figure=fig,
        style={"height": "700px", "marginTop": 30}
    )
], fluid=True)

@app.callback(
    Output('proches-output', 'children'),
    Input('circo-dropdown', 'value')
)
def update_proches(selected_idx):
    if selected_idx is None or selected_idx == '':
        return ""
    idx = int(selected_idx)
    # Correction : s'assurer que les valeurs sont bien de type float
    pca_mat = pivot[['PCA1', 'PCA2']].astype(float).values
    ref = pivot.loc[idx, ['PCA1', 'PCA2']].astype(float).values
    dists = np.linalg.norm(pca_mat - ref, axis=1)
    dists[idx] = np.inf
    proches_idx = np.argsort(dists)[:3]
    rows = []
    for i in proches_idx:
        row = pivot.iloc[i]
        rows.append({
            "Code département": row['Code du département'],
            "Circonscription": row['Code de la circonscription'],
            "Département": row['departmentName'],
            "Région": row['regionName'],
            "Distance": f"{dists[i]:.2f}"
        })
    return dash_table.DataTable(
        columns=[{"name": k, "id": k} for k in rows[0].keys()],
        data=rows,
        style_cell={"textAlign": "center", "fontSize": 16},
        style_header={"backgroundColor": "#f5e9d7", "fontWeight": "bold"},
        style_data={"backgroundColor": "#fff"},
        style_table={"marginTop": 20, "marginBottom": 20, "width": "80%", "marginLeft": "auto", "marginRight": "auto"}
    )

if __name__ == "__main__":
    app.run(debug=True)
