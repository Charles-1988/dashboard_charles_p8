import os
import json
import boto3
import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import requests

# ======================================================
# ENV VARIABLES
# ======================================================

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
MAIN_CLIENTS_FILE = os.environ["CLIENTS_MAIN_FILE"]
COMPARE_CLIENTS_FILE = os.environ["CLIENTS_COMPARE_FILE"]
API_PREDICT_URL = os.environ["API_PREDICT_URL"]

# ======================================================
# S3 CONNECTION
# ======================================================

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def load_clients_from_s3(filename, sample_size=300):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
    data = json.loads(obj["Body"].read().decode("utf-8"))
    df = pd.DataFrame(data)

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    df.index = df.index.astype(str)
    return df

# ======================================================
# LOAD DATA ONCE (IMPORTANT FOR MEMORY)
# ======================================================

df1 = load_clients_from_s3(MAIN_CLIENTS_FILE)
df2 = load_clients_from_s3(COMPARE_CLIENTS_FILE)
df = pd.concat([df1, df2])

if "TARGET" not in df.columns:
    raise ValueError("La colonne TARGET est manquante")

top_features = [c for c in df.columns if c != "TARGET"]

# ======================================================
# API CALL
# ======================================================

def call_api(payload):
    try:
        res = requests.post(API_PREDICT_URL, json=payload, timeout=10)
        return res.json()
    except:
        return None

# ======================================================
# DASH APP
# ======================================================

app = dash.Dash(__name__)
server = app.server
app.title = "Dashboard Crédit"

app.layout = html.Div([

    html.H1("Dashboard Scoring Crédit", style={"textAlign":"center"}),

    html.Label("Choisir un client"),
    dcc.Dropdown(
        id="select-client",
        options=[{"label": idx, "value": idx} for idx in df.index],
        value=df.index[0]
    ),

    dcc.Tabs(id="tabs", value="risk", children=[

        dcc.Tab(label="Risque", value="risk"),

        dcc.Tab(label="Distribution TARGET", value="dist"),

        dcc.Tab(label="Scatter", value="scatter", children=[
            html.Br(),
            html.Label("Variable X"),
            dcc.Dropdown(
                id="scatter-x",
                options=[{"label": f, "value": f} for f in top_features],
                value=top_features[0]
            ),
            html.Label("Variable Y"),
            dcc.Dropdown(
                id="scatter-y",
                options=[{"label": f, "value": f} for f in top_features],
                value=top_features[1]
            ),
            dcc.Graph(id="scatter-graph")
        ]),

        dcc.Tab(label="SHAP", value="shap")

    ]),

    html.Div(id="tabs-content")

])

# ======================================================
# SCATTER CALLBACK
# ======================================================

@app.callback(
    Output("scatter-graph","figure"),
    Input("scatter-x","value"),
    Input("scatter-y","value")
)
def update_scatter(x, y):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color="TARGET",
        opacity=0.5
    )
    return fig

# ======================================================
# MAIN TAB CALLBACK
# ======================================================

@app.callback(
    Output("tabs-content","children"),
    Input("tabs","value"),
    Input("select-client","value")
)
def update_tab(tab, client_id):

    if not client_id:
        return html.Div("Aucun client sélectionné")

    client_data = df.loc[client_id].to_dict()
    payload = {f: float(client_data[f]) for f in top_features}

    # ---------------- RISK TAB
    if tab == "risk":

        result = call_api(payload)
        proba = result.get("proba", 0) * 100 if result else 0

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba,
            title={'text': "Probabilité défaut (%)"},
            gauge={'axis': {'range': [0, 100]}}
        ))

        return dcc.Graph(figure=fig)

    # ---------------- DISTRIBUTION TARGET
    elif tab == "dist":

        fig = px.histogram(df, x="TARGET")
        return dcc.Graph(figure=fig)

    # ---------------- SHAP
    elif tab == "shap":

        result = call_api(payload)

        if result and "shap_values" in result:
            shap_values = result["shap_values"]

            fig = px.bar(
                x=list(shap_values.values()),
                y=list(shap_values.keys()),
                orientation="h"
            )
            return dcc.Graph(figure=fig)

        return html.Div("SHAP indisponible")

    return html.Div()

# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)