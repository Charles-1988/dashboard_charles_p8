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

# Variables d'environnement
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
MAIN_CLIENTS_FILE = os.environ["CLIENTS_MAIN_FILE"]
COMPARE_CLIENTS_FILE = os.environ["CLIENTS_COMPARE_FILE"]
API_PREDICT_URL = os.environ.get("API_PREDICT_URL")
API_EXPLAIN_URL = os.environ.get("API_EXPLAIN_URL")

# Connexion S3
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Fonctions S3
def load_clients_from_s3(filename, features=None, nrows=None):
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
    if nrows is not None:
        df = df.head(nrows)
    df["source_file"] = filename
    return df

def save_clients_to_s3(df, filename):
    data = df.drop(columns=["source_file"], errors="ignore").to_dict(orient="records")
    s3.put_object(Bucket=BUCKET_NAME, Key=filename,
                  Body=json.dumps(data), ContentType="application/json")

# Appel API
def call_api(url, payload):
    if not url:
        return None
    try:
        res = requests.post(url, json=payload, timeout=10).json()
        return res if "error" not in res else None
    except:
        return None

# Préparation des données
def convert_days_birth(value):
    return int(-value / 365)

def prepare_client_value(feat, value):
    return convert_days_birth(value) if feat=="DAYS_BIRTH" else value

def prepare_dataframe(df, feat_list, sample_size=5000):
    df_copy = df.copy()
    for feat in feat_list:
        if feat=="DAYS_BIRTH":
            df_copy[feat] = df_copy[feat].apply(convert_days_birth)
    if len(df_copy) > sample_size:
        df_copy = df_copy.sample(n=sample_size, random_state=42)
    return df_copy

# Chargement des clients
sample_df = load_clients_from_s3(MAIN_CLIENTS_FILE, nrows=5)
top_features = [f for f in sample_df.columns if f != "TARGET"]

clients_df_main = load_clients_from_s3(MAIN_CLIENTS_FILE, features=top_features, nrows=5000)
clients_df_extra = load_clients_from_s3(COMPARE_CLIENTS_FILE, features=top_features, nrows=5000)
clients_df = pd.concat([clients_df_main, clients_df_extra])
df_compar = clients_df.copy()

# Dash App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Dashboard Crédit"

# Layout
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

# Callback Tabs
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

    # Tab Gauge
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
        children = [html.H3(score_text, style={"textAlign":"center"}),
                    dcc.Loading(dcc.Graph(figure=fig_gauge), type="circle")]
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
            children.append(dcc.Loading(dcc.Graph(figure=fig_shap), type="circle"))

        return html.Div(children)

    # Tab Distribution
    if tab=="tab-distrib":
        return html.Div([
            html.Label("Choisir une feature :"),
            dcc.Dropdown(id="select-feature-distrib",
                         options=[{"label": f, "value": f} for f in top_features],
                         value=top_features[0],
                         clearable=False),
            html.Div(id="output-hist-distrib")
        ])

    # Tab Scatter
    if tab=="tab-scatter":
        return html.Div([
            html.Label("Choisir 2 features :"),
            dcc.Dropdown(id="select-feature-scatter-x",
                         options=[{"label": f, "value": f} for f in top_features],
                         value=top_features[0],
                         clearable=False),
            dcc.Dropdown(id="select-feature-scatter-y",
                         options=[{"label": f, "value": f} for f in top_features],
                         value=top_features[1],
                         clearable=False),
            html.Div(id="output-scatter")
        ])

    # Tab Ajouter client
    if tab=="tab-add-client":
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

# Callback Distribution
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
    df_plot_sample = df_plot.sample(n=min(5000,len(df_plot)), random_state=42)

    def make_hist(data, title, color):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=50, opacity=0.7, marker_color=color))
        fig.add_vline(x=client_val, line_color="blue", line_width=3,
                      annotation_text="Client", annotation_position="top")
        fig.update_layout(title=title, xaxis_title=feature, yaxis_title="Nombre de clients",
                          width=900, height=500)
        return fig

    return html.Div([
        dcc.Loading(dcc.Graph(figure=make_hist(df_plot_sample[feature], f"Distribution globale de {feature}", "royalblue")), type="circle"),
        dcc.Loading(dcc.Graph(figure=make_hist(df_plot_sample[df_plot_sample["TARGET"]==0][feature], f"Crédit accordé (0) pour {feature}", "green")), type="circle"),
        dcc.Loading(dcc.Graph(figure=make_hist(df_plot_sample[df_plot_sample["TARGET"]==1][feature], f"Crédit refusé (1) pour {feature}", "red")), type="circle")
    ])

# Callback Scatter
@app.callback(
    Output("output-scatter","children"),
    [Input("select-feature-scatter-x","value"),
     Input("select-feature-scatter-y","value"),
     Input("select-client","value")]
)
def update_scatter(x_feat, y_feat, selected_client):
    client_data = clients_df.loc[selected_client].to_dict()
    df_graph = prepare_dataframe(clients_df, [x_feat, y_feat], sample_size=5000)
    client_x = prepare_client_value(x_feat, client_data[x_feat])
    client_y = prepare_client_value(y_feat, client_data[y_feat])

    fig_scatter = px.scatter(df_graph,
                             x=x_feat, y=y_feat, opacity=0.3, title=f"{x_feat} vs {y_feat}")
    fig_scatter.add_scatter(x=[client_x], y=[client_y], mode="markers",
                            marker=dict(size=14,color="red"), name="Client")
    fig_scatter.add_vline(x=df_graph[x_feat].median(), line_dash="dash", line_color="black")
    fig_scatter.add_hline(y=df_graph[y_feat].median(), line_dash="dash", line_color="black")
    fig_scatter.update_layout(width=900, height=600)

    return dcc.Loading(dcc.Graph(figure=fig_scatter), type="circle")

# Callback Ajouter client
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

# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)