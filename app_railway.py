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

# ==============================
# ENV VARIABLES
# ==============================

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
MAIN_CLIENTS_FILE = os.environ["CLIENTS_MAIN_FILE"]
COMPARE_CLIENTS_FILE = os.environ["CLIENTS_COMPARE_FILE"]
API_PREDICT_URL = os.environ.get("API_PREDICT_URL")
API_EXPLAIN_URL = os.environ.get("API_EXPLAIN_URL")

# ==============================
# S3 CONNECTION
# ==============================

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def load_clients_from_s3(filename, nrows=None):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
    data = json.loads(obj["Body"].read().decode("utf-8"))
    if isinstance(data, dict):
        data = list(data.values())
    df = pd.DataFrame(data)
    if nrows:
        df = df.sample(n=min(nrows, len(df)), random_state=42)
    return df

def save_clients_to_s3(df, filename):
    data = df.to_dict(orient="records")
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=filename,
        Body=json.dumps(data),
        ContentType="application/json"
    )

# ==============================
# INITIAL LOAD (IMPORTANT)
# ==============================

df_main_init = load_clients_from_s3(MAIN_CLIENTS_FILE, nrows=100)
df_extra_init = load_clients_from_s3(COMPARE_CLIENTS_FILE, nrows=100)
clients_df_init = pd.concat([df_main_init, df_extra_init])

top_features = [f for f in clients_df_init.columns if f != "TARGET"]

# ==============================
# API CALL
# ==============================

def call_api(url, payload):
    if not url:
        return None
    try:
        res = requests.post(url, json=payload, timeout=10)
        return res.json()
    except:
        return None

# ==============================
# DASH APP
# ==============================

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Dashboard Crédit"

app.layout = html.Div([
    html.H1("Dashboard Crédit", style={"textAlign":"center"}),

    html.Label("Choisir un client :"),
    dcc.Dropdown(id="select-client"),

    dcc.Tabs(id="tabs-dashboard", value="tab-gauge", children=[
        dcc.Tab(label="Risque", value="tab-gauge"),
        dcc.Tab(label="Distribution", value="tab-distrib"),
        dcc.Tab(label="Scatter", value="tab-scatter"),
        dcc.Tab(label="Nouveau client", value="tab-add-client")
    ]),

    html.Div(id="tabs-content")
])

# ==============================
# INIT DROPDOWN
# ==============================

@app.callback(
    Output("select-client","options"),
    Output("select-client","value"),
    Input("tabs-dashboard","value")
)
def init_clients(_):
    df_main = load_clients_from_s3(MAIN_CLIENTS_FILE, nrows=50)
    df_extra = load_clients_from_s3(COMPARE_CLIENTS_FILE, nrows=50)
    df = pd.concat([df_main, df_extra])
    df.index = df.index.astype(str)

    options = [{"label": idx, "value": idx} for idx in df.index]
    value = df.index[0] if len(df) > 0 else None
    return options, value

# ==============================
# MAIN TABS
# ==============================

@app.callback(
    Output("tabs-content","children"),
    Input("tabs-dashboard","value"),
    Input("select-client","value")
)
def update_tabs(tab, selected_client):

    if not selected_client:
        return html.Div("Aucun client disponible.")

    df_main = load_clients_from_s3(MAIN_CLIENTS_FILE, nrows=200)
    df_extra = load_clients_from_s3(COMPARE_CLIENTS_FILE, nrows=200)
    df = pd.concat([df_main, df_extra])
    df.index = df.index.astype(str)

    client_data = df.loc[selected_client].to_dict()
    payload = {f: float(client_data[f]) for f in top_features}

    if tab == "tab-gauge":
        res = call_api(API_PREDICT_URL, payload)
        proba = res.get("proba",0)*100 if res else 0

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba,
            title={'text': "Probabilité de défaut (%)"},
            gauge={'axis': {'range':[0,100]}}
        ))

        return dcc.Graph(figure=fig)

    elif tab == "tab-distrib":
        feature = top_features[0]
        fig = px.histogram(df, x=feature)
        return dcc.Graph(figure=fig)

    elif tab == "tab-scatter":
        fig = px.scatter(df, x=top_features[0], y=top_features[1])
        return dcc.Graph(figure=fig)

    elif tab == "tab-add-client":
        median_vals = df.median()
        return html.Div([
            html.Button("Ajouter", id="add-client-button"),
            html.Br(),
            dcc.Input(id="new-client-name", value="Nouveau client"),
            html.Br(),
            *[
                html.Div([
                    html.Label(f),
                    dcc.Input(id=f"input-{f}", type="number", value=float(median_vals[f]))
                ])
                for f in top_features
            ]
        ])

# ==============================
# ADD CLIENT
# ==============================

@app.callback(
    Output("select-client","options"),
    Output("select-client","value"),
    Input("add-client-button","n_clicks"),
    State("new-client-name","value"),
    *[State(f"input-{f}","value") for f in top_features]
)
def add_client(n_clicks, name, *vals):

    df_main = load_clients_from_s3(MAIN_CLIENTS_FILE, nrows=5000)
    df_extra = load_clients_from_s3(COMPARE_CLIENTS_FILE, nrows=5000)
    df = pd.concat([df_main, df_extra])

    if n_clicks and name:
        new_row = pd.DataFrame([dict(zip(top_features, vals))], index=[name])
        df = pd.concat([new_row, df])
        save_clients_to_s3(df, MAIN_CLIENTS_FILE)

    df.index = df.index.astype(str)
    options = [{"label": idx, "value": idx} for idx in df.index]
    value = name if n_clicks and name else df.index[0]

    return options, value

# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)