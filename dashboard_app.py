import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go


clients_df = pd.read_json("data/Tous_les_clients.json", orient="index")
top_features = list(clients_df.columns)
if 'index' in top_features:
    top_features.remove('index')


df_compar = pd.read_json("data/clients_target.json", orient="index")  

API_PREDICT_URL = "https://credit-scoring-api-tqja.onrender.com/predict"
API_EXPLAIN_URL = "https://credit-scoring-api-tqja.onrender.com/explain"


def call_api(url, payload):
    try:
        res = requests.post(url, json=payload).json()
        if "error" in res:
            print(f"Erreur API {url}: {res['error']}")
            return None
        return res
    except Exception as e:
        print(f"Erreur API {url}: {e}")
        return None

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


app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Dashboard Crédit"


app.layout = html.Div([
    html.H1("Dashboard Crédit", style={"textAlign":"center"}),

    html.Div([  
        html.Label("Choisir un client :"),
        dcc.Dropdown(
            id="select-client",
            options=[{"label": idx, "value": idx} for idx in clients_df.index],
            value=clients_df.index[0]
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


@app.callback(
    Output("tabs-content", "children"),
    [Input("tabs-dashboard", "value"),
     Input("select-client", "value")]
)
def update_tabs(tab, selected_client):
    if not selected_client:
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

        
        color_bar = "green" if proba < 0.9 else "red"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba*100,
            title={'text': "Probabilité de défaut (%)"},
            gauge={
                'axis': {'range':[0,100]},
                'bar': {'color': 'black'},
                'steps': [
                    {'range':[0,9],'color':'green'},
                    {'range':[9,100],'color':'red'}
                ]
            }
        ))

        shap_res = call_api(API_EXPLAIN_URL, client_data)
        if shap_res:
            shap_vals = [shap_res[f] for f in top_features]
            shap_df = pd.DataFrame({"Feature": top_features, "SHAP": shap_vals})
            shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values(ascending=False).index).head(10)

            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                x=shap_df["SHAP"],
                y=shap_df["Feature"],
                orientation='h',
                marker_color=['red' if v>0 else 'blue' for v in shap_df["SHAP"]]
            ))
            fig_shap.update_layout(
                title="Top 10 SHAP features",
                xaxis_title="SHAP value",
                yaxis=dict(autorange="reversed"),
                width=900,
                height=500
            )

            return html.Div([
                html.H3(score_text, style={"textAlign":"center"}),
                dcc.Graph(figure=fig_gauge),
                dcc.Graph(figure=fig_shap)
            ])
        else:
            return html.Div([
                html.H3(score_text, style={"textAlign":"center"}),
                dcc.Graph(figure=fig_gauge)
            ])

    elif tab=="tab-distrib":
        return html.Div([
            html.Label("Choisir une feature pour comparer le client :"),
            dcc.Dropdown(
                id="select-feature-distrib",
                options=[{"label": f, "value": f} for f in top_features],
                value=top_features[0],
                clearable=False
            ),
            html.Div(id="output-hist-distrib")
        ])

    elif tab=="tab-scatter":
        return html.Div([
            html.Label("Choisir 2 features pour scatter plot :"),
            dcc.Dropdown(
                id="select-feature-scatter-x",
                options=[{"label": f, "value": f} for f in top_features],
                value=top_features[0],
                clearable=False
            ),
            dcc.Dropdown(
                id="select-feature-scatter-y",
                options=[{"label": f, "value": f} for f in top_features],
                value=top_features[1],
                clearable=False
            ),
            html.Div(id="output-scatter")
        ])

    elif tab=="tab-add-client":
        median_vals = clients_df.median()
        return html.Div([
            html.Button("Ajouter un client", id="add-client-button", n_clicks=0),
            html.Br(), html.Br(),
            html.Label("Nom du nouveau client :"),
            dcc.Input(id="new-client-name", type="text", value="Nouveau client"),
            html.Br(), html.Br(),
            *[html.Div([html.Label(f), dcc.Input(id=f"input-{f}", type="number",
                value=float(median_vals[f]))]) for f in top_features]
        ])


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

   
    fig_all = go.Figure()
    fig_all.add_trace(go.Histogram(
        x=df_plot[feature],
        nbinsx=50,
        name="Tous les clients",
        opacity=0.5,
        marker_color="royalblue"
    ))
    fig_all.add_vline(x=client_val, line_color="blue", line_width=3,
                      annotation_text="Client", annotation_position="top")
    fig_all.update_layout(title=f"Position du client dans la distribution globale de {feature}",
                          xaxis_title=feature, yaxis_title="Nombre de clients", legend_title="Légende",
                          width=900, height=500)

    fig_target0 = go.Figure()
    fig_target0.add_trace(go.Histogram(
        x=df_plot[df_plot["TARGET"]==0][feature],
        nbinsx=50,
        name="Crédit accordé (0)",
        opacity=0.7,
        marker_color="green"
    ))
    fig_target0.add_vline(x=client_val, line_color="blue", line_width=3,
                          annotation_text="Client", annotation_position="top")
    fig_target0.update_layout(title=f"Position du client parmi les crédits accordés (0) pour {feature}",
                              xaxis_title=feature, yaxis_title="Nombre de clients", legend_title="Légende",
                              width=900, height=500)

    fig_target1 = go.Figure()
    fig_target1.add_trace(go.Histogram(
        x=df_plot[df_plot["TARGET"]==1][feature],
        nbinsx=50,
        name="Crédit refusé (1)",
        opacity=0.7,
        marker_color="red"
    ))
    fig_target1.add_vline(x=client_val, line_color="blue", line_width=3,
                          annotation_text="Client", annotation_position="top")
    fig_target1.update_layout(title=f"Position du client parmi les crédits refusés (1) pour {feature}",
                              xaxis_title=feature, yaxis_title="Nombre de clients", legend_title="Légende",
                              width=900, height=500)

    return html.Div([
        dcc.Graph(figure=fig_all),
        dcc.Graph(figure=fig_target0),
        dcc.Graph(figure=fig_target1)
    ])


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

    fig_scatter = px.scatter(
        df_graph.sample(n=min(5000,len(df_graph)), random_state=42),
        x=x_feat, y=y_feat, opacity=0.3, title=f"{x_feat} vs {y_feat}"
    )
    fig_scatter.add_scatter(x=[client_x], y=[client_y], mode="markers",
                            marker=dict(size=14,color="red"), name="Client")
    fig_scatter.add_vline(x=df_graph[x_feat].median(), line_dash="dash", line_color="black",
                          annotation_text="Médiane X", annotation_position="top left")
    fig_scatter.add_hline(y=df_graph[y_feat].median(), line_dash="dash", line_color="black",
                          annotation_text="Médiane Y", annotation_position="top right")
    fig_scatter.update_layout(width=900, height=600)

    return dcc.Graph(figure=fig_scatter)


@app.callback(
    Output("select-client","options"),
    [Input("add-client-button","n_clicks")],
    [State("new-client-name","value")] +
    [State(f"input-{f}","value") for f in top_features]
)
def add_client(n_clicks, name, *vals):
    global clients_df
    
    if n_clicks > 0 and name:
        vals_dict = {f: v if v is not None else float(clients_df[f].median()) 
                     for f, v in zip(top_features, vals)}
        
        new_row = pd.DataFrame([vals_dict], index=[name])
        
        
        clients_df = pd.concat([new_row, clients_df])
    
    return [{"label": idx, "value": idx} for idx in clients_df.index]


if __name__ == "__main__":
    app.run(debug=True)























