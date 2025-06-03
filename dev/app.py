# Libraries
import pandas as pd
import numpy as np
from skforecast.datasets import fetch_dataset
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import shap
from skforecast.plot import set_dark_theme
from sklearn.inspection import PartialDependenceDisplay
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import base64
import io

# Data Preparation and Modeling
def prepare_data():
    data = fetch_dataset(name="vic_electricity")
    data = data.resample('D').agg({'Demand': 'sum', 'Temperature': 'mean'})
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    return data

def train_forecaster(data_train):
    window_features = RollingFeatures(stats=['mean'], window_sizes=24)
    exog_features = ['Temperature', 'day_of_week', 'month']
    forecaster = ForecasterRecursive(
        regressor=LGBMRegressor(random_state=123, verbose=-1),
        lags=7,
        window_features=window_features
    )
    forecaster.fit(y=data_train['Demand'], exog=data_train[exog_features])
    return forecaster, exog_features

def backtest_and_explain(forecaster, data, exog_features, end_train):
    cv = TimeSeriesFold(steps=24, initial_train_size=len(data.loc[:end_train]))
    _, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=data['Demand'],
        exog=data[exog_features],
        cv=cv,
        metric='mean_absolute_error',
        return_predictors=True,
    )

    explainer = shap.TreeExplainer(forecaster.regressor)
    predictors = predictions.iloc[:, 2:]
    shap_values = explainer(predictors)

    # Map SHAP values to their corresponding timestamps
    shap_dict = {predictions.index[i]: shap_values[i] for i in range(len(predictions))}
    return predictions, shap_dict

# Plot creation functions
def create_main_figure(data_test, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data_test.index, y=data_test['Demand'],
        name="Test", mode="lines",
        line=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Demand: %{y}<extra>Test Data</extra>'
    ))
    fig.add_trace(go.Scatter(
        x=predictions.index, y=predictions['pred'],
        name="Prediction", mode="lines+markers",
        line=dict(color='orange'),
        hovertemplate='Date: %{x}<br>Prediction: %{y}<extra>Forecast</extra>'
    ))
    fig.update_layout(
        title="Real value vs predicted in test data",
        xaxis_title="Date",
        yaxis_title="Demand",
        width=900,
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def matplotlib_to_base64(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def generate_shap_plot(shap_value):
    shap.plots.waterfall(shap_value, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    img = matplotlib_to_base64(fig, dpi=150)
    plt.close(fig)
    return img

# Prepare everything before app starts
data = prepare_data()
end_train = '2014-12-01 23:59:00'
data_train = data.loc[:end_train]
data_test = data.loc[end_train:]

forecaster, exog_features = train_forecaster(data_train)
predictions, shap_values = backtest_and_explain(forecaster, data, exog_features, end_train)

fig_main = create_main_figure(data_test, predictions)

# Dash App
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Forecast Explainability with SHAP"),
    dcc.Graph(id='main-plot', figure=fig_main, clear_on_unhover=True),
    html.Div(id='shap-output', style={'marginTop': '15px', 'fontWeight': 'bold'}, children="Click on a prediction point to view SHAP values"),
    html.Div([
        html.Img(id='shap-plot', style={'width': '80%', 'maxWidth': '900px', 'height': 'auto', 'border': '1px solid #ccc', 'marginTop': '10px'}),
    ], style={'textAlign': 'center'}),
    html.Div([
        html.Button("Reset", id='reset-btn', n_clicks=0, style={'marginTop': '20px'}),
        dcc.Loading(id="loading", type="circle", children=html.Div(id="loading-output")),
    ], style={'textAlign': 'center'})
], style={'maxWidth': '1000px', 'margin': 'auto', 'padding': '20px'})

@app.callback(
    Output('shap-plot', 'src'),
    Output('shap-output', 'children'),
    Input('main-plot', 'clickData'),
    Input('reset-btn', 'n_clicks'),
    prevent_initial_call=True
)
def update_shap(clickData, reset_clicks):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'reset-btn':
        return None, "Click on a prediction point to view SHAP values"

    if clickData is None:
        return None, dash.no_update

    if clickData['points'][0]['curveNumber'] != 1:
        return None, "Please click on the prediction line (orange line)."

    clicked_time = pd.to_datetime(clickData['points'][0]['x'])
    shap_val = shap_values.get(clicked_time)

    if shap_val is None:
        return None, f"No SHAP value available for {clicked_time}"

    img = generate_shap_plot(shap_val)
    return img, f"SHAP values for prediction on {clicked_time}"

if __name__ == '__main__':
    app.run(debug=True)
