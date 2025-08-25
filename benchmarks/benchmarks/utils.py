################################################################################
#                               Benchmarking Common                            #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.express.colors import qualitative
from packaging.version import parse, Version


def plot_benchmark_results(df, function_name, add_median=True, add_mean=True):
    """
    Plot benchmark results with jittered strip points and per-point standard deviation error bars,
    with error bars matching point colors.
    """

    np.random.seed(42164)
    sorted_versions = sorted(df['skforecast_version'].unique(), key=parse)
    df['skforecast_version'] = pd.Categorical(df['skforecast_version'], categories=sorted_versions, ordered=True)
    df = df.sort_values("skforecast_version")
    version_to_num = {v: i for i, v in enumerate(sorted_versions)}
    df['x_jittered'] = df['skforecast_version'].map(version_to_num).astype(float) + np.random.uniform(-0.05, 0.05, size=len(df))
    version_colors = {
        v: qualitative.Plotly[i % len(qualitative.Plotly)] 
        for i, v in enumerate(sorted_versions)
    }
    platform_symbols = {
        pltf: symbol for pltf, symbol in zip(df['platform'].unique(), [
            'circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star'
        ])
    }

    fig = go.Figure()
    for version in sorted_versions:
        for pltf, symbol in platform_symbols.items():
            sub_df = df[(df['skforecast_version'] == version) & (df['platform'] == pltf)]
            if sub_df.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub_df['x_jittered'],
                    y=sub_df['run_time_avg'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=version_colors[version],
                        symbol=symbol,
                        opacity=0.7
                    ),
                    error_y=dict(
                        type='data',
                        array=sub_df['run_time_std_dev'],
                        visible=True,
                        color=version_colors[version],
                        thickness=1.5,
                        width=5
                    ),
                    name=f'{version} - {pltf}',
                    text=sub_df.apply(lambda row: (
                        f"Forecaster: {row['forecaster_name']}<br>"
                        f"Regressor: {row['regressor_name']}<br>"
                        f"Function: {row['function_name']}<br>"
                        f"Function_hash: {row['function_hash']}<br>"
                        f"Datetime: {row['datetime']}<br>"
                        f"Python version: {row['python_version']}<br>"
                        f"skforecast version: {row['skforecast_version']}<br>"
                        f"numpy version: {row['numpy_version']}<br>"
                        f"pandas version: {row['pandas_version']}<br>"
                        f"sklearn version: {row['sklearn_version']}<br>"
                        f"lightgbm version: {row['lightgbm_version']}<br>"
                        f"Platform: {row['platform']}<br>"
                        f"Processor: {row['processor']}<br>"
                        f"CPU count: {row['cpu_count']}<br>"
                        f"Memory (GB): {row['memory_gb']:.2f}<br>"
                        f"Run time avg: {row['run_time_avg']:.4f} seconds<br>"
                        f"Run time std dev: {row['run_time_std_dev']:.4f} seconds"
                    ), axis=1),
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=True
                )
            )

    if add_median:
        medians = df.groupby('skforecast_version', observed=True)['run_time_avg'].median().reset_index()
        medians['x_center'] = medians['skforecast_version'].map(version_to_num)
        fig.add_trace(
            go.Scatter(
                x=medians['x_center'],
                y=medians['run_time_avg'],
                mode='lines+markers',
                line=dict(color='black', width=2),
                marker=dict(size=8, color='black'),
                name='Median (per version)'
            )
        )

    if add_mean:
        means = df.groupby('skforecast_version', observed=True)['run_time_avg'].mean().reset_index()
        means['x_center'] = means['skforecast_version'].map(version_to_num)
        fig.add_trace(
            go.Scatter(
                x=means['x_center'],
                y=means['run_time_avg'],
                mode='lines+markers',
                line=dict(color='black', width=2, dash='dash'),
                marker=dict(size=8, color='black'),
                name='Mean (per version)'
            )
        )

    fig.update_layout(
        title=f"Execution time of {function_name}",
        xaxis=dict(
            tickmode='array',
            tickvals=list(version_to_num.values()),
            ticktext=list(version_to_num.keys()),
            title='skforecast version',
            tickangle=-45
        ),
        yaxis_title='Execution time (seconds)',
        showlegend=True
    )
    fig.show()


def plot_benchmark_results_v2(df, forecaster=None, regressor=None, add_median=True, add_mean=True):
    """
    Plot benchmark results with jittered strip points and per-point standard deviation error bars,
    with error bars matching point colors.
    """
    if forecaster is not None:
        if not isinstance(forecaster, list):
            forecaster = [forecaster]
        df = df.query("forecaster_name in @forecaster")

    if regressor is not None:
        if not isinstance(regressor, list):
            regressor = [regressor]
        df = df.query("regressor_name in @regressor")

    df['skforecast_version'] = pd.Categorical(
        df['skforecast_version'], 
        ordered=True, 
        categories=sorted(df['skforecast_version'].unique())
    )


    def try_version(v):
        try: 
            return Version(str(v).lstrip('v'))
        except: 
            return Version("0")  # fallback


    versions_sorted = sorted(df['skforecast_version'].unique(), key=try_version)
    df['skforecast_version'] = pd.Categorical(
        df['skforecast_version'], 
        categories=versions_sorted, 
        ordered=True
    )

    version_colors = {
        v: qualitative.Plotly[i % len(qualitative.Plotly)] 
        for i, v in enumerate(versions_sorted)
    }

    methods = list(df['function_name'].unique())
    metrics = [
        ("Mean", "run_time_avg"),
        ("Median", "run_time_median"),
        ("p95", "run_time_p95"),
        ("Std", "run_time_std")
    ]

    fig = go.Figure()

    # Creamos un trace por (método, métrica). Solo mostramos (método0, media) al inicio.
    for j, (label, col) in enumerate(metrics):
        for i, m in enumerate(methods):
            sub_df = df[df['function_name'] == m].sort_values('skforecast_version')
            if sub_df.empty:
                continue
            visible = (i == 0 and j == 0)  # solo la primera combinación visible
            # Sólo ponemos barras de error para la media (std)
            error_y = dict(
                type='data', 
                array=sub_df['run_time_std'], 
                visible=(label in ['Mean', 'Median'])
            )
            fig.add_trace(go.Scatter(
                x=sub_df['skforecast_version'],
                y=sub_df[col],
                mode='lines+markers',
                name=f"{m} — {label}",
                visible=visible,
                hovertemplate="<b>%{x}</b><br>"
                            f"Método: {m}<br>"
                            f"Métrica: {label}<br>"
                            "Tiempo: %{y:.6f} s<br>"
                            "runs: %{customdata}",
                customdata=np.c_[sub_df['cpu_count']],
                error_y=error_y,
            ))

    # Helpers para visibilidades
    def mask_for(method_idx, metric_idx):
        """Devuelve un vector de visibilidad para todos los traces."""
        vis = [False] * (len(methods) * len(metrics))
        # índice lineal = metric_idx * len(methods) + method_idx
        vis[metric_idx * len(methods) + method_idx] = True
        return vis

    # Dropdown 1: método
    buttons_methods = []
    for i, m in enumerate(methods):
        buttons_methods.append(dict(
            label=m,
            method="update",
            args=[{"visible": mask_for(i, 0)},  # por defecto deja 'Media'
                {"title": f"Tiempos por versión — método: {m}"}]
        ))

    # Dropdown 2: métrica
    buttons_metrics = []
    for j, (label, _) in enumerate(metrics):
        # Por simplicidad, cambiamos a la métrica para el método actualmente activo
        # (Plotly no “recuerda” cuál, así que usamos active method=0 como base;
        # se sincroniza al cambiar método por el usuario).
        buttons_metrics.append(dict(
            label=label,
            method="update",
            args=[{"visible": mask_for(0, j)},
                {}]
        ))

    # Botones Linear/Log
    buttons_scale = [
        dict(label="Linear", method="relayout", args=[{"yaxis.type": "linear"}]),
        dict(label="Log",    method="relayout", args=[{"yaxis.type": "log"}]),
    ]

    fig.update_layout(
        title=f"Tiempos por versión — método: {methods[0]}",
        xaxis_title="Versión del paquete",
        yaxis_title="Tiempo (s)",
        template="plotly_white",
        updatemenus=[
            # 1) Botones Linear/Log (arriba‑derecha)
            dict(
                type="buttons",
                buttons=buttons_scale,
                showactive=True,
                direction="left",
                x=1.00, y=1.18, xanchor="right", yanchor="top",
                pad={"r": 0, "t": 0}
            ),
            # 2) Dropdown de Métrica
            dict(
                type="dropdown",
                buttons=buttons_metrics,
                showactive=True,
                direction="down",
                x=0.91, y=1.18, xanchor="right", yanchor="top",
                pad={"r": 0, "t": 0}
            ),
            # 3) Dropdown de Método
            dict(
                type="dropdown",
                buttons=buttons_methods,
                showactive=True,
                direction="down",
                x=0.83, y=1.18, xanchor="right", yanchor="top",
                pad={"r": 0, "t": 0}
            ),
        ],
        margin=dict(l=60, r=20, t=80, b=60),  # un poco más de margen superior
        legend=dict(title=""),
    )
    fig.update_xaxes(tickangle=0)  # cambia a 45 si las versiones son largas

    # Sincroniza el dropdown de métricas con el método visible al cambiarlo
    # (workaround simple: al hacer click en un método, queda 'Media';
    # si quieres sincronización perfecta, hay que añadir callbacks JS o
    # construir todas las combinaciones de visibilidad en los menús).
    fig.show()
