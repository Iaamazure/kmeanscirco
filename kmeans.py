import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
import numpy as np
import os
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

files = [
    'data_a_utiliser/leg_2017.csv',
    'data_a_utiliser/leg_2022.csv',
    'data_a_utiliser/leg_2024.csv'
]
frames = []
for f in files:
    df = pd.read_csv(f)
    year = f.split('_')[-1].split('.')[0]
    df['ANNEE'] = year
    frames.append(df)
data = pd.concat(frames, ignore_index=True)
id_cols = ['Code du département', 'Code de la circonscription', 'ANNEE']
feature_cols = [col for col in data.columns if col not in id_cols]
X = data[feature_cols].fillna(0)
id_circo = ['Code du département', 'Code de la circonscription']
pivot = data.pivot_table(
    index=id_circo,
    columns='ANNEE',
    values=[col for col in data.columns if col not in id_cols],
    fill_value=0
)
pivot.columns = [f"{score}_{annee}" for score, annee in pivot.columns]
pivot = pivot.reset_index()
X = pivot.drop(columns=id_circo)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
pivot['cluster'] = clusters
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
plt.title('Clustering of constituencies')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend(title='Cluster')
plt.tight_layout()
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
    title='Clustering of French legislative constituencies with KMeans',
    labels={'PCA1': 'Principal component 1', 'PCA2': 'Principal component 2'}
)
fig.update_traces(marker=dict(size=18, opacity=0.95, line=dict(width=1, color='black')))
regions = pd.read_csv('data_a_utiliser/departments_regions_france_2016.csv', dtype=str)
regions['departmentCode'] = regions['departmentCode'].str.lower()
regions['departmentName'] = regions['departmentName'].str.strip()
regions['regionName'] = regions['regionName'].str.strip()
def format_dep(dep):
    dep = str(dep).lower()
    if dep.isdigit():
        return dep.zfill(2)
    return dep
pivot['dep_code_str'] = pivot['Code du département'].apply(format_dep)
pivot = pivot.merge(regions, left_on='dep_code_str', right_on='departmentCode', how='left')
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
    title='<b>Clustering of French legislative constituencies (KMeans, PCA 2D)</b>',
    labels={
        'PCA1': '',
        'PCA2': '',
        'departmentName': 'Department',
        'regionName': 'Region',
        'Code du département': 'Department code',
        'Code de la circonscription': 'Constituency',
        'cluster': 'Cluster'
    },
    template='plotly_white',
    opacity=0.95
)
fig.update_traces(
    marker=dict(size=12, opacity=0.95, line=dict(width=1, color='black')),
    selector=dict(mode='markers')
)
fig.update_layout(
    font=dict(family='Arial', size=15),
    title_font=dict(size=22, family='Arial Black'),
    legend=dict(title='Cluster', font=dict(size=14)),
    xaxis=dict(showgrid=True, zeroline=False, showticklabels=False, gridcolor='#e5e5e5'),
    yaxis=dict(showgrid=True, zeroline=False, showticklabels=False, gridcolor='#e5e5e5'),
    plot_bgcolor='#FFF8E7',
    paper_bgcolor='#FFF8E7'
)
fig.for_each_trace(lambda t: t.update(hovertemplate=t.hovertemplate.replace('=', ':')))
print(f"KMeans inertia: {kmeans.inertia_:.2f}")
sil_score = silhouette_score(X_scaled, clusters)
print(f"Silhouette score: {sil_score:.3f} (close to 1 = good clusters, close to 0 = overlap)")
random_labels = np.random.randint(0, n_clusters, size=len(X_scaled))
sil_score_random = silhouette_score(X_scaled, random_labels)
print(f"Silhouette score (random clustering): {sil_score_random:.3f}")
def find_closest_constituencies(num_departement, num_circo, n=3):
    def format_dep(dep):
        dep = str(dep).lower()
        if dep.isdigit():
            return dep.zfill(2)
        return dep
    dep_code = format_dep(num_departement)
    circo_code = str(num_circo)
    mask = (pivot['dep_code_str'] == dep_code) & (pivot['Code de la circonscription'].astype(str) == circo_code)
    if not mask.any():
        print("Constituency not found.")
        return
    idx = pivot[mask].index[0]
    dists = np.linalg.norm(pivot[['PCA1', 'PCA2']].values - pivot.loc[idx, ['PCA1', 'PCA2']].values, axis=1)
    dists[idx] = np.inf
    closest_idx = np.argsort(dists)[:n]
    print(f"The {n} closest constituencies to {num_departement}-{num_circo} are:")
    for i in closest_idx:
        row = pivot.iloc[i]
        print(f"  - {row['Code du département']}-{row['Code de la circonscription']} ({row['departmentName']}, {row['regionName']}) | Distance: {dists[i]:.2f}")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
pivot['circo_search'] = pivot['Code du département'].astype(str) + '-' + pivot['Code de la circonscription'].astype(str) + \
    ' (' + pivot['departmentName'] + ', ' + pivot['regionName'] + ')'
explanation_lines = [
    "This project lets you explore a clustering of French legislative constituencies based on the results of the first rounds of the 2017, 2022, and 2024 legislative elections. The k-means method is used to group constituencies with similar voting patterns. Each color represents a group. These groups do not have an official meaning, but you can guess some common points:",
    "- Dark purple: pro-RN",
    "- Orange-red: pro-Macron",
    "- Purple: pro-left",
    "",
    "This project offers 3 tools:",
    "- A search bar to find the constituencies most similar to the one you select.",
    "- A graphical visualization of the different groups. Each point is a constituency. Hover for details. The axes have no particular meaning, they just help visualize the groups.",
    "- A map visualization of the different groups.",
    "",
    "By the way, I am really sorry that it looks so ugly. This is my first project of this kind, I am still learning the tools!"
]
explanation_message = []
for i, line in enumerate(explanation_lines):
    if line == "This project offers 3 tools:":
        explanation_message.append(html.Br())
    if line == "This project offers 3 tools:":
        explanation_message.append(html.Br())
    if line == "By the way, I am really sorry that it looks so ugly. This is my first project of this kind, I am still learning the tools!":
        explanation_message.append(html.Br())
        explanation_message.append(html.Br())
    if line.startswith("-"):
        explanation_message.append(html.Br())
    explanation_message.append(line)
app.layout = dbc.Container([
    html.Div([
        html.P(explanation_message, style={"fontSize": 18, "backgroundColor": "#FFF8E7", "padding": "18px", "borderRadius": "8px", "marginTop": 25, "marginBottom": 25})
    ]),
    html.H2("Find similar constituencies", style={"marginTop": 10}),
    dcc.Dropdown(
        id='circo-dropdown',
        options=[{'label': str(c), 'value': str(i)} for i, c in enumerate(pivot['circo_search'])],
        placeholder="Choose a constituency...",
        style={"width": "100%", "marginBottom": 20}
    ),
    html.Div(id='proches-output'),
    dcc.Graph(
        id='scatter-graph',
        figure=fig,
        style={"height": "700px", "marginTop": 30}
    ),
    html.Hr(),
    html.H3("Map of constituencies by cluster", style={"marginTop": 30}),
    dcc.Graph(
        id='map-graph',
        figure={},
        style={"height": "700px", "marginTop": 10}
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
    pca_mat = pivot[['PCA1', 'PCA2']].astype(float).values
    ref = pivot.loc[idx, ['PCA1', 'PCA2']].astype(float).values
    dists = np.linalg.norm(pca_mat - ref, axis=1)
    dists[idx] = np.inf
    closest_idx = np.argsort(dists)[:3]
    rows = []
    for i in closest_idx:
        row = pivot.iloc[i]
        rows.append({
            "Department code": row['Code du département'],
            "Constituency": row['Code de la circonscription'],
            "Department": row['departmentName'],
            "Region": row['regionName'],
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
try:
    import geopandas as gpd
    circo_gdf = gpd.read_file('data_a_utiliser/circonscriptions-legislatives-p10.geojson')
    circo_gdf['Code du département'] = circo_gdf['codeDepartement'].astype(str).str.lower().str.zfill(2)
    circo_gdf['Code de la circonscription'] = circo_gdf['codeCirconscription'].astype(str).str.lower().str.zfill(2)
    circo_gdf['id_circo'] = circo_gdf['Code du département'] + '-' + circo_gdf['Code de la circonscription']
    pivot['Code du département'] = pivot['Code du département'].astype(str).str.lower().str.zfill(2)
    pivot['Code de la circonscription'] = pivot['Code de la circonscription'].astype(str).str.zfill(2)
    pivot['id_circo'] = pivot['Code du département'].str.zfill(2) + '-' + pivot['Code du département'].str.zfill(2) + pivot['Code de la circonscription'].str.zfill(2)
    geojson = circo_gdf.__geo_interface__
    for i, feature in enumerate(geojson['features']):
        feature['id'] = circo_gdf.iloc[i]['id_circo']
    ids_geojson = set(circo_gdf['id_circo'])
    ids_pivot = set(pivot['id_circo'])
    cluster_palette = px.colors.qualitative.Plotly[:pivot['cluster'].nunique()]
    map_fig = px.choropleth_mapbox(
        pivot,
        geojson=geojson,
        locations='id_circo',
        color='cluster',
        color_continuous_scale=None,
        color_discrete_sequence=cluster_palette,
        hover_name='circo_label',
        hover_data={'Code du département': True, 'Code de la circonscription': True, 'departmentName': True, 'regionName': True},
        mapbox_style='carto-positron',
        zoom=4.5,
        center={"lat": 46.6, "lon": 2.5},
        opacity=0.7,
        title="Map of constituencies colored by cluster (surface)"
    )
    map_fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
except ImportError:
    map_fig = go.Figure()
    map_fig.update_layout(title="Map display impossible: geopandas not installed")
except Exception as e:
    map_fig = go.Figure()
    map_fig.update_layout(title=f"Error displaying map: {e}")
app.layout = dbc.Container([
    html.Div([
        html.P(explanation_message, style={"fontSize": 18, "backgroundColor": "#FFF8E7", "padding": "18px", "borderRadius": "8px", "marginTop": 25, "marginBottom": 25})
    ]),
    html.H2("Find similar constituencies", style={"marginTop": 10}),
    dcc.Dropdown(
        id='circo-dropdown',
        options=[{'label': str(c), 'value': str(i)} for i, c in enumerate(pivot['circo_search'])],
        placeholder="Choose a constituency...",
        style={"width": "100%", "marginBottom": 20}
    ),
    html.Div(id='proches-output'),
    dcc.Graph(
        id='scatter-graph',
        figure=fig,
        style={"height": "700px", "marginTop": 30}
    ),
    html.Hr(),
    html.H3("Map of constituencies by cluster", style={"marginTop": 30}),
    dcc.Graph(
        id='map-graph',
        figure=map_fig,
        style={"height": "700px", "marginTop": 10}
    )
], fluid=True)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
