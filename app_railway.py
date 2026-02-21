import traceback
import os
import json
import boto3
import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import requests
import plotly.express as px
import plotly.graph_objects as go

# -----------------------
# Vérification des variables d'environnement
# -----------------------
required_env_vars = [
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "BUCKET_NAME",
    "CLIENTS_MAIN_FILE", "CLIENTS_COMPARE_FILE"
]

for var in required_env_vars:
    if not os.environ.get(var):
        raise EnvironmentError(f"La variable d'environnement {var} n'est pas définie !")

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
MAIN_CLIENTS_FILE = os.environ["CLIENTS_MAIN_FILE"]
COMPARE_CLIENTS_FILE = os.environ["CLIENTS_COMPARE_FILE"]

API_PREDICT_URL = os.environ.get("API_PREDICT_URL")
API_EXPLAIN_URL = os.environ.get("API_EXPLAIN_URL")

if not API_PREDICT_URL:
    print("⚠️ API_PREDICT_URL non défini ! Les prédictions ne fonctionneront pas.")
if not API_EXPLAIN_URL:
    print("⚠️ API_EXPLAIN_URL non défini ! Les explications SHAP ne fonctionneront pas.")

# -----------------------
# Connexion S3
# -----------------------
try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    # Vérification rapide du bucket
    files = s3.list_objects_v2(Bucket=BUCKET_NAME)
    if 'Contents' in files:
        print("Fichiers dans le bucket :", [f["Key"] for f in files['Contents']])
    else:
        print("⚠️ Aucun fichier trouvé dans le bucket !")
except Exception as e:
    print("Erreur lors de la connexion S3 :")
    traceback.print_exc()
    raise

# -----------------------
# Fonctions S3
# -----------------------
def load_clients_from_s3(filename, nrows=None):
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
        data = json.load(obj["Body"])  
        # Si c'est une dict, on prend ses valeurs
        if isinstance(data, dict):
            data = list(data.values())
        # Si nrows est défini, on prend juste les premières lignes
        if nrows is not None:
            data = data[:nrows]
        df = pd.DataFrame(data)
        df["source_file"] = filename
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {filename} depuis S3 :")
        traceback.print_exc()
        return pd.DataFrame()  # Retourne un DataFrame vide en cas d'erreur
    
def save_clients_to_s3(df, filename):
    try:
        data = df.drop(columns=["source_file"], errors="ignore").to_dict(orient="records")
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=filename,
            Body=json.dumps(data),
            ContentType="application/json"
        )
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier {filename} sur S3 :")
        traceback.print_exc()

# -----------------------
# Chargement clients
# -----------------------
clients_df_main = load_clients_from_s3(MAIN_CLIENTS_FILE, nrows=5000)
clients_df_extra = load_clients_from_s3(COMPARE_CLIENTS_FILE, nrows=5000)
clients_df = pd.concat([clients_df_main, clients_df_extra])
df_compar = clients_df.copy()

# Liste des features
top_features = [f for f in clients_df.columns if f != "index"]

# -----------------------
# Fonctions API
# -----------------------
def call_api(url, payload):
    if not url:
        return None
    try:
        res = requests.post(url, json=payload).json()
        if "error" in res:
            print(f"Erreur API {url}: {res['error']}")
            return None
        return res
    except Exception as e:
        print(f"Erreur API {url}: {e}")
        return None

# -----------------------
# Fonctions utilitaires
# -----------------------
def convert_days_birth(value):
    return int(-value / 365)

def prepare_client_value(feat, value):
    return convert_days_birth(value) if feat=="DAYS_BIRTH" else value

def prepare_dataframe(df, feat_list):
    df_copy = df.copy()
    for feat in feat_list:
        if feat=="DAYS_BIRTH":
            df_copy[feat] = df_copy[feat].apply(convert_days_birth)
    return df_copy

# -----------------------
# Dash App
# -----------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Dashboard Crédit"

# -----------------------
# Layout
# -----------------------
app.layout = html.Div([
    html.H1("Dashboard Crédit", style={"textAlign":"center"}),
    html.Div([  
        html.Label("Choisir un client :"),
        dcc.Dropdown(
            id="select-client",
            options=[{"label": idx, "value": idx} for idx in clients_df.index],
            value=clients_df.index[0] if len(clients_df) > 0 else None
        )
    ], style={"width":"40%","margin":"auto"}),
    html.Br(),
    dcc.Tabs(id="tabs-dashboard", value="tab-gauge", children=[
        dcc.Tab(label="Risque de défaut", value="tab-gauge"),
        dcc.Tab(label="Comparaison client", value="tab-distrib"),
        dcc.Tab(label="Comparaison 2 features", value="tab-scatter"),
        dcc.Tab(label="Nouveau client", value="tab-add-client")
    ]),
    html.Div(id="tabs-content", style={"marginTop":20})
])

# -----------------------
# Callback Tabs
# -----------------------
@app.callback(
    Output("tabs-content", "children"),
    [Input("tabs-dashboard", "value"),
     Input("select-client", "value")]
)
def update_tabs(tab, selected_client):
    if selected_client is None:
        return html.Div("Sélectionnez un client")

    client_data = clients_df.loc[selected_client].to_dict()
    client_data = {feat: float(client_data.get(feat, 0)) for feat in top_features}

    if tab=="tab-gauge":
        res = call_api(API_PREDICT_URL, client_data)
        if res:
            proba = res.get("proba",0)
            classe = res.get("classe",0)
            decision = "Crédit ACCORDÉ ✅" if classe==0 else "Crédit REFUSÉ ❌"
            score_text = f"{selected_client} — Probabilité de défaut : {proba*100:.1f}% — {decision}"
        else:
            proba = 0
            score_text = "Erreur API predict"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba*100,
            title={'text': "Probabilité de défaut (%)"},
            gauge={'axis': {'range':[0,100]},
                   'bar': {'color': 'black'},
                   'steps':[{'range':[0,90],'color':'green'},
                            {'range':[90,100],'color':'red'}]}
        ))

        shap_res = call_api(API_EXPLAIN_URL, client_data)
        if shap_res:
            shap_vals = [shap_res.get(f,0) for f in top_features]
            shap_df = pd.DataFrame({"Feature": top_features, "SHAP": shap_vals})
            shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values(ascending=False).index).head(10)

            fig_shap = go.Figure(go.Bar(
                x=shap_df["SHAP"],
                y=shap_df["Feature"],
                orientation='h',
                marker_color=['red' if v>0 else 'blue' for v in shap_df["SHAP"]]
            ))
            fig_shap.update_layout(title="Top 10 SHAP features",
                                   xaxis_title="SHAP value",
                                   yaxis=dict(autorange="reversed"),
                                   width=900, height=500)

            return html.Div([html.H3(score_text, style={"textAlign":"center"}),
                             dcc.Graph(figure=fig_gauge),
                             dcc.Graph(figure=fig_shap)])
        else:
            return html.Div([html.H3(score_text, style={"textAlign":"center"}),
                             dcc.Graph(figure=fig_gauge)])

    elif tab=="tab-distrib":
        return html.Div([html.Label("Choisir une feature pour comparer le client :"),
                         dcc.Dropdown(id="select-feature-distrib",
                                      options=[{"label": f, "value": f} for f in top_features],
                                      value=top_features[0],
                                      clearable=False),
                         html.Div(id="output-hist-distrib")])

    elif tab=="tab-scatter":
        return html.Div([html.Label("Choisir 2 features pour scatter plot :"),
                         dcc.Dropdown(id="select-feature-scatter-x",
                                      options=[{"label": f, "value": f} for f in top_features],
                                      value=top_features[0],
                                      clearable=False),
                         dcc.Dropdown(id="select-feature-scatter-y",
                                      options=[{"label": f, "value": f} for f in top_features],
                                      value=top_features[1],
                                      clearable=False),
                         html.Div(id="output-scatter")])

    elif tab=="tab-add-client":
        median_vals = clients_df.median()
        return html.Div([
            html.Button("Ajouter un client", id="add-client-button", n_clicks=0),
            html.Br(), html.Br(),
            html.Label("Nom du nouveau client :"),
            dcc.Input(id="new-client-name", type="text", value="Nouveau client"),
            html.Br(), html.Br(),
            *[html.Div([html.Label(f), dcc.Input(id=f"input-{f}", type="number",
                                                 value=float(median_vals[f]))])
              for f in top_features]
        ])

# -----------------------
# Callback Distribution
# -----------------------
@app.callback(
    Output("output-hist-distrib","children"),
    [Input("select-feature-distrib","value"),
     Input("select-client","value")]
)
def update_hist(feature, selected_client):
    client_val = prepare_client_value(feature, clients_df.loc[selected_client][feature])
    df_plot = df_compar.copy()
    if feature=="DAYS_BIRTH":
        df_plot[feature] = df_plot[feature].apply(convert_days_birth)

    # Distribution globale
    fig_all = go.Figure()
    fig_all.add_trace(go.Histogram(x=df_plot[feature], nbinsx=50, name="Tous les clients",
                                   opacity=0.5, marker_color="royalblue"))
    fig_all.add_vline(x=client_val, line_color="blue", line_width=3,
                      annotation_text="Client", annotation_position="top")
    fig_all.update_layout(title=f"Distribution globale de {feature}",
                          xaxis_title=feature, yaxis_title="Nombre de clients",
                          width=900, height=500)

    # Crédit accordé
    fig_target0 = go.Figure()
    fig_target0.add_trace(go.Histogram(x=df_plot[df_plot["TARGET"]==0][feature], nbinsx=50,
                                       name="Crédit accordé (0)", opacity=0.7, marker_color="green"))
    fig_target0.add_vline(x=client_val, line_color="blue", line_width=3,
                          annotation_text="Client", annotation_position="top")
    fig_target0.update_layout(title=f"Crédit accordé (0) pour {feature}",
                              xaxis_title=feature, yaxis_title="Nombre de clients",
                              width=900, height=500)

    # Crédit refusé
    fig_target1 = go.Figure()
    fig_target1.add_trace(go.Histogram(x=df_plot[df_plot["TARGET"]==1][feature], nbinsx=50,
                                       name="Crédit refusé (1)", opacity=0.7, marker_color="red"))
    fig_target1.add_vline(x=client_val, line_color="blue", line_width=3,
                          annotation_text="Client", annotation_position="top")
    fig_target1.update_layout(title=f"Crédit refusé (1) pour {feature}",
                              xaxis_title=feature, yaxis_title="Nombre de clients",
                              width=900, height=500)

    return html.Div([dcc.Graph(figure=fig_all),
                     dcc.Graph(figure=fig_target0),
                     dcc.Graph(figure=fig_target1)])

# -----------------------
# Callback Scatter
# -----------------------
@app.callback(
    Output("output-scatter","children"),
    [Input("select-feature-scatter-x","value"),
     Input("select-feature-scatter-y","value"),
     Input("select-client","value")]
)
def update_scatter(x_feat, y_feat, selected_client):
    client_data = clients_df.loc[selected_client].to_dict()
    df_graph = prepare_dataframe(clients_df, [x_feat, y_feat])
    client_x = prepare_client_value(x_feat, client_data[x_feat])
    client_y = prepare_client_value(y_feat, client_data[y_feat])

    fig_scatter = px.scatter(df_graph.sample(n=min(5000,len(df_graph)), random_state=42),
                             x=x_feat, y=y_feat, opacity=0.3, title=f"{x_feat} vs {y_feat}")
    fig_scatter.add_scatter(x=[client_x], y=[client_y], mode="markers",
                            marker=dict(size=14,color="red"), name="Client")
    fig_scatter.add_vline(x=df_graph[x_feat].median(), line_dash="dash", line_color="black",
                          annotation_text="Médiane X", annotation_position="top left")
    fig_scatter.add_hline(y=df_graph[y_feat].median(), line_dash="dash", line_color="black",
                          annotation_text="Médiane Y", annotation_position="top right")
    fig_scatter.update_layout(width=900, height=600)

    return dcc.Graph(figure=fig_scatter)

# -----------------------
# Callback Ajouter client
# -----------------------
@app.callback(
    [Output("select-client","options"),
     Output("select-client","value")],
    [Input("add-client-button","n_clicks")],
    [State("new-client-name","value")] + [State(f"input-{f}","value") for f in top_features]
)
def add_client(n_clicks, name, *vals):
    global clients_df
    if n_clicks > 0 and name:
        vals_dict = {f: v if v is not None else float(clients_df[f].median()) 
                     for f, v in zip(top_features, vals)}
        new_row = pd.DataFrame([vals_dict], index=[name])
        clients_df = pd.concat([new_row, clients_df])
        save_clients_to_s3(clients_df, MAIN_CLIENTS_FILE)

    options = [{"label": idx, "value": idx} for idx in clients_df.index]
    selected = name if n_clicks > 0 else clients_df.index[0]
    return options, selected

# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8050))
        app.run_server(debug=True, host="0.0.0.0", port=port)
    except Exception as e:
        print("Erreur au démarrage du serveur :")
        traceback.print_exc()
        raise