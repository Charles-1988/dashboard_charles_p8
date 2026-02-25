import os
import json
import boto3
import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import requests
import logging
from functools import lru_cache

# --- Logging pour debug ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Variables d'environnement ---
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
MAIN_CLIENTS_FILE = os.environ["CLIENTS_MAIN_FILE"]
COMPARE_CLIENTS_FILE = os.environ["CLIENTS_COMPARE_FILE"]
API_PREDICT_URL = os.environ.get("API_PREDICT_URL")
API_EXPLAIN_URL = os.environ.get("API_EXPLAIN_URL")

# --- Connexion S3 ---
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# --- Fonctions S3 ---
def load_clients_from_s3(filename, features=None, nrows=None):
    """Charge les données depuis S3, échantillonnées si besoin."""
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
    data = json.loads(obj["Body"].read().decode("utf-8"))
    if isinstance(data, dict):
        data = list(data.values())
    df = pd.DataFrame(data)
    
    # Optimisation types pandas
    for col in df.select_dtypes('float'):
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes('int'):
        df[col] = df[col].astype('int32')
    for col in df.select_dtypes('object'):
        df[col] = df[col].astype('category')
    
    if features:
        keep_cols = [f for f in features if f in df.columns]
        if "TARGET" in df.columns:
            keep_cols.append("TARGET")
        df = df[keep_cols]
    
    if nrows and len(df) > nrows:
        df = df.sample(n=nrows, random_state=42)
    
    df["source_file"] = filename
    return df

def load_client_by_index(filename, idx, features=None):
    """Charge uniquement un client précis pour minimiser la RAM."""
    df = load_clients_from_s3(filename, features=features, nrows=None)
    return df.loc[[idx]] if idx in df.index else pd.DataFrame()

def save_clients_to_s3(df, filename):
    data = df.drop(columns=["source_file"], errors="ignore").to_dict(orient="records")
    s3.put_object(Bucket=BUCKET_NAME, Key=filename,
                  Body=json.dumps(data), ContentType="application/json")

# --- API ---
@lru_cache(maxsize=128)
def call_api_cached(url, payload_str):
    """Cache simple pour limiter les appels API répétitifs."""
    payload = json.loads(payload_str)
    if not url:
        return None
    try:
        res = requests.post(url, json=payload, timeout=10).json()
        return res if "error" not in res else None
    except Exception as e:
        logger.error("API call failed: %s", e)
        return None

def call_api(url, payload):
    return call_api_cached(json.dumps(payload), json.dumps(payload))

# --- Préparation des données ---
def convert_days_birth(value):
    return int(-value / 365)

def prepare_dataframe(df, feat_list, sample_size=500):
    df_copy = df.copy()
    for feat in feat_list:
        if feat=="DAYS_BIRTH":
            df_copy[feat] = df_copy[feat].apply(convert_days_birth)
    if len(df_copy) > sample_size:
        df_copy = df_copy.sample(n=sample_size, random_state=42)
    return df_copy

# --- Dash App ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Dashboard Crédit"

# --- Précharger uniquement les index pour dropdown ---
df_main_idx = load_clients_from_s3(MAIN_CLIENTS_FILE, nrows=200)
df_extra_idx = load_clients_from_s3(COMPARE_CLIENTS_FILE, nrows=200)
df_clients_index = pd.concat([df_main_idx, df_extra_idx])
client_options = [{"label": idx, "value": idx} for idx in df_clients_index.index]

# --- Layout ---
app.layout = html.Div([
    html.H1("Dashboard Crédit", style={"textAlign":"center"}),
    html.Div([
        html.Label("Choisir un client :"),
        dcc.Dropdown(id="select-client", options=client_options,
                     value=client_options[0]['value'] if client_options else None)
    ], style={"width":"40%","margin":"auto"}),
    dcc.Store(id='store-client-data'),  # pour stocker les données du client
    html.Br(),
    dcc.Tabs(id="tabs-dashboard", value="tab-gauge", children=[
        dcc.Tab(label="Risque de défaut", value="tab-gauge"),
        dcc.Tab(label="Comparaison client", value="tab-distrib"),
        dcc.Tab(label="Comparaison 2 features", value="tab-scatter"),
        dcc.Tab(label="Nouveau client", value="tab-add-client")
    ]),
    html.Div(id="tabs-content", style={"marginTop":20})
])

# --- Stocker les données du client sélectionné ---
@app.callback(
    Output("store-client-data","data"),
    Input("select-client","value")
)
def load_selected_client(selected_client):
    if not selected_client:
        return {}
    df_main = load_client_by_index(MAIN_CLIENTS_FILE, selected_client)
    df_extra = load_client_by_index(COMPARE_CLIENTS_FILE, selected_client)
    df = pd.concat([df_main, df_extra])
    if df.empty:
        return {}
    data = df.iloc[0].to_dict()
    return data

# --- Contenu des tabs ---
@app.callback(
    Output("tabs-content","children"),
    Input("tabs-dashboard","value"),
    State("store-client-data","data")
)
def update_tabs(tab, client_data):
    if not client_data:
        return html.Div("Aucun client disponible.")

    top_features = [f for f in client_data.keys() if f != "TARGET"]

    # --- Tab Risque ---
    if tab=="tab-gauge":
        res = call_api(API_PREDICT_URL, client_data)
        proba = res.get("proba",0) if res else 0
        classe = res.get("classe",0) if res else 0
        decision = "Crédit ACCORDÉ ✅" if classe==0 else "Crédit REFUSÉ ❌"
        score_text = f"{client_data.get('client_name','Client')} — Probabilité de défaut : {proba*100:.1f}% — {decision}" if res else "Erreur API predict"

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
        children = [html.H3(score_text, style={"textAlign":"center"}), dcc.Graph(figure=fig_gauge)]
        if shap_res:
            shap_vals = [shap_res.get(f,0) for f in top_features]
            shap_df = pd.DataFrame({"Feature": top_features, "SHAP": shap_vals})
            shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values(ascending=False).index).head(10)
            fig_shap = go.Figure(go.Bar(
                x=shap_df["SHAP"], y=shap_df["Feature"], orientation='h',
                marker_color=['red' if v>0 else 'blue' for v in shap_df["SHAP"]]
            ))
            fig_shap.update_layout(title="Top 10 SHAP features", xaxis_title="SHAP value",
                                   yaxis=dict(autorange="reversed"), width=900, height=500)
            children.append(dcc.Graph(figure=fig_shap))
        return html.Div(children)

    # --- Tab Distribution ---
    elif tab=="tab-distrib":
        df_sample = prepare_dataframe(pd.DataFrame([client_data]), top_features, sample_size=200)
        return html.Div([
            html.Label("Choisir une feature :"),
            dcc.Dropdown(id="select-feature-distrib",
                         options=[{"label": f, "value": f} for f in top_features],
                         value=top_features[0]),
            html.Div(id="output-hist-distrib")
        ])

    # --- Tab Scatter ---
    elif tab=="tab-scatter":
        df_sample = prepare_dataframe(pd.DataFrame([client_data]), top_features, sample_size=200)
        return html.Div([
            html.Label("Choisir 2 features :"),
            dcc.Dropdown(id="select-feature-scatter-x",
                         options=[{"label": f, "value": f} for f in top_features],
                         value=top_features[0]),
            dcc.Dropdown(id="select-feature-scatter-y",
                         options=[{"label": f, "value": f} for f in top_features],
                         value=top_features[1]),
            html.Div(id="output-scatter")
        ])

    # --- Tab Ajouter client ---
    elif tab=="tab-add-client":
        median_vals = pd.DataFrame([client_data]).median()
        return html.Div([
            html.Button("Ajouter un client", id="add-client-button", n_clicks=0),
            html.Br(), html.Br(),
            html.Label("Nom du nouveau client :"),
            dcc.Input(id="new-client-name", type="text", value="Nouveau client"),
            html.Br(), html.Br(),
            *[html.Div([html.Label(f), dcc.Input(id=f"input-{f}", type="number", value=float(median_vals[f]))])
              for f in top_features]
        ])

# --- Run server sur Render ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    logger.info(f"Starting Dash app on port {port}")
    app.run_server(debug=False, host="0.0.0.0", port=port)