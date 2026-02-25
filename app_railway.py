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
def load_clients_from_s3(filename, features=None, nrows=500):
    """Charge les données depuis S3 de façon sécurisée et échantillonnée."""
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
    data = json.loads(obj["Body"].read().decode("utf-8"))
    if isinstance(data, dict):
        data = list(data.values())
    df = pd.DataFrame(data)
    if features:
        keep_cols = [f for f in features if f in df.columns]
        if "TARGET" in df.columns:
            keep_cols.append("TARGET")
        df = df[keep_cols]
    if nrows is not None and len(df) > nrows:
        df = df.sample(n=min(nrows,len(df)), random_state=42)
    df["source_file"] = filename
    return df

def save_clients_to_s3(df, filename):
    data = df.drop(columns=["source_file"], errors="ignore").to_dict(orient="records")
    s3.put_object(Bucket=BUCKET_NAME, Key=filename,
                  Body=json.dumps(data), ContentType="application/json")

# --- API ---
def call_api(url, payload):
    if not url:
        return None
    try:
        res = requests.post(url, json=payload, timeout=10).json()
        return res if "error" not in res else None
    except Exception as e:
        logger.error("API call failed: %s", e)
        return None

# --- Préparation des données ---
def convert_days_birth(value):
    return int(-value / 365)

def prepare_client_value(feat, value):
    return convert_days_birth(value) if feat=="DAYS_BIRTH" else value

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

# --- Layout ---
app.layout = html.Div([
    html.H1("Dashboard Crédit", style={"textAlign":"center"}),
    html.Div([
        html.Label("Choisir un client :"),
        dcc.Dropdown(id="select-client")
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

# --- Initialisation du dropdown ---
@app.callback(
    Output("select-client","options"),
    Output("select-client","value"),
    Input("tabs-dashboard","value")
)
def init_clients(_):
    df_main = load_clients_from_s3(MAIN_CLIENTS_FILE, nrows=500)
    df_extra = load_clients_from_s3(COMPARE_CLIENTS_FILE, nrows=500)
    df_combined = pd.concat([df_main, df_extra])
    options = [{"label": idx, "value": idx} for idx in df_combined.index]
    value = df_combined.index[0] if len(df_combined) > 0 else None
    return options, value

# --- Contenu des tabs ---
@app.callback(
    Output("tabs-content","children"),
    [Input("tabs-dashboard","value"),
     Input("select-client","value")]
)
def update_tabs(tab, selected_client):
    if selected_client is None:
        return html.Div("Aucun client disponible.")

    df_main = load_clients_from_s3(MAIN_CLIENTS_FILE, nrows=500)
    df_extra = load_clients_from_s3(COMPARE_CLIENTS_FILE, nrows=500)
    clients_df = pd.concat([df_main, df_extra])
    top_features = [f for f in clients_df.columns if f != "TARGET"]

    client_data = clients_df.loc[selected_client].to_dict()
    client_data = {feat: float(client_data[feat]) for feat in top_features}

    # --- Tab Risque ---
    if tab=="tab-gauge":
        res = call_api(API_PREDICT_URL, client_data)
        proba = res.get("proba",0) if res else 0
        classe = res.get("classe",0) if res else 0
        decision = "Crédit ACCORDÉ ✅" if classe==0 else "Crédit REFUSÉ ❌"
        score_text = f"{selected_client} — Probabilité de défaut : {proba*100:.1f}% — {decision}" if res else "Erreur API predict"

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
        return html.Div([
            html.Label("Choisir une feature :"),
            dcc.Dropdown(id="select-feature-distrib",
                         options=[{"label": f, "value": f} for f in top_features],
                         value=top_features[0]),
            html.Div(id="output-hist-distrib")
        ])

    # --- Tab Scatter ---
    elif tab=="tab-scatter":
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
        median_vals = clients_df.median()
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