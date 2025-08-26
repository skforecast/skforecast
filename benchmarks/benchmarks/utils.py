################################################################################
#                               Benchmarking Common                            #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License      #
################################################################################
# coding=utf-8

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.express.colors import qualitative
from packaging.version import parse, Version, InvalidVersion


def plot_benchmark_results(
    df: pd.DataFrame, 
    forecaster_names: str | list[str], 
    regressors: str | list[str] | None = None, 
    add_median: bool = True, 
    add_mean: bool = True
) -> None:
    """
    Plot interactive benchmark results by method and package version.

    This function renders an interactive Plotly chart that visualizes execution
    times for multiple methods across `skforecast` versions. Each data point
    represents a run (or an aggregated run) for a given `(method, version)` and is
    jittered horizontally to avoid overlap. Points are **colored by version** and
    (optionally) per-version **median** and **mean** lines are overlaid for the
    selected method. A dropdown allows switching the visible method.

    Parameters
    ----------
    df : pandas DataFrame
        Input data to benchmark.
    forecaster_names : str, list 
        Forecaster(s) to filter in `df` (column `forecaster_name`).
    regressors : str, list, default None
        Regressor(s) to filter in `df` (column `regressor_name`). If `None`,
        no additional filtering by regressor is applied.
    add_median : bool, default True
        If `True`, draw one per-version **median** line for the selected method.
    add_mean : bool, default True
        If `True`, draw one per-version **mean** line for the selected method.
    
    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure object containing the benchmark results.

    """

    if not isinstance(forecaster_names, list):
        forecaster_names = [forecaster_names]
    df = df.query("forecaster_name in @forecaster_names")

    if regressors is not None:
        if not isinstance(regressors, list):
            regressors = [regressors]
        df = df.query("regressor_name in @regressors")

    def try_version(v):
        try: 
            return Version(str(v).lstrip('v'))
        except InvalidVersion: 
            return Version("0")  # fallback

    versions_sorted = sorted(df['skforecast_version'].unique(), key=try_version)
    version_to_num = {v: i for i, v in enumerate(versions_sorted)}
    df['skforecast_version'] = pd.Categorical(
        df['skforecast_version'], 
        categories=versions_sorted, 
        ordered=True
    )
    df['x_jittered'] = (
        df['skforecast_version'].map(version_to_num).astype(float) + 
        np.random.uniform(-0.05, 0.05, size=len(df))
    )

    # --- paleta por versión ---
    version_colors = {
        v: qualitative.Plotly[i % len(qualitative.Plotly)] 
        for i, v in enumerate(versions_sorted)
    }

    methods = list(df['method_name'].unique())

    fig = go.Figure()

    # --- un trace por (método, métrica); colores por versión en los puntos ---
    method_to_traces = {m: [] for m in methods}
    for i, m in enumerate(methods):
        for version in versions_sorted:

            sub_df = df[
                (df['method_name'] == m) & 
                (df['skforecast_version'] == version)
            ]
            if sub_df.empty:
                continue

            marker = dict(
                size=10,
                color=version_colors[version],
                opacity=0.85,
                line=dict(color="white", width=1)
            )   

            error_y = dict(
                type='data', 
                array=sub_df['run_time_std'].to_numpy(), 
                visible=not sub_df['run_time_std'].isna().all(),
                color=version_colors[version],
                thickness=1.5,
                width=5
            )

            fig.add_trace(go.Scatter(
                x=sub_df['x_jittered'],
                y=sub_df['run_time_avg'],
                mode='markers',
                marker=marker,
                error_y=error_y,
                visible=(i == 0),
                # name=f"{methods[i]} — {label}",
                text = sub_df.apply(lambda row: (
                    f"Forecaster: {row['forecaster_name']}<br>"
                    f"Regressor: {row['regressor_name']}<br>"
                    f"Function: {row['function_name']}<br>"
                    f"Function_hash: {row['function_hash']}<br>"
                    f"Method: {row['method_name']}<br>"
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
                    f"Run time avg: {row['run_time_avg']:.6f} s<br>"
                    f"Run time median: {row['run_time_median']:.6f} s<br>"
                    f"Run time p95: {row['run_time_p95']:.6f} s<br>"
                    f"Run time std: {row['run_time_std']:.6f} s<br>"
                    f"Nº repeats: {row['n_repeats']}"
                ), axis=1),
                hovertemplate = '%{text}<extra></extra>'
            ))
            method_to_traces[m].append(len(fig.data) - 1)

    median_trace_id = {}
    if add_median:
        for i, m in enumerate(methods):

            med_df = (
                df[df['method_name'] == m]
                .groupby('skforecast_version', observed=True)['run_time_avg']
                .median()
                .reset_index()
            )
            if med_df.empty:
                continue

            median_color = "#374151"
            med_df['x_center'] = med_df['skforecast_version'].map(version_to_num)
            fig.add_trace(go.Scatter(
                x=med_df['x_center'],
                y=med_df['run_time_avg'],
                mode='lines+markers',
                line=dict(color=median_color, width=2),
                marker=dict(size=8, color=median_color),
                name='Median (per version)',
                visible=(i == 0)
            ))
            median_trace_id[m] = len(fig.data) - 1

    mean_trace_id = {}
    if add_mean:
        for i, m in enumerate(methods):

            mean_df = (
                df[df['method_name'] == m]
                .groupby('skforecast_version', observed=True)['run_time_avg']
                .mean()
                .reset_index()
            )
            if mean_df.empty:
                continue

            mean_color = "#9CA3AF"
            mean_df['x_center'] = mean_df['skforecast_version'].map(version_to_num)
            fig.add_trace(go.Scatter(
                x=mean_df['x_center'],
                y=mean_df['run_time_avg'],
                mode='lines+markers',
                line=dict(color=mean_color, width=2, dash='dash'),
                marker=dict(size=8, color=mean_color),
                name='Mean (per version)',
                visible=(i == 0)
            ))
            mean_trace_id[m] = len(fig.data) - 1

    def visible_mask_for(method):
        n = len(fig.data)
        mask = [False] * n

        # puntos del método (todas sus versiones)
        for idx in method_to_traces.get(method, []):
            mask[idx] = True
        # su mediana y media (si existen)
        if add_median and method in median_trace_id:
            mask[median_trace_id[method]] = True
        if add_mean and method in mean_trace_id:
            mask[mean_trace_id[method]] = True

        return mask

    buttons_methods = []
    for i, m in enumerate(methods):
        buttons_methods.append(dict(
            label=m,
            method="update",
            args=[
                {"visible": visible_mask_for(m)}, 
                {"title": f"Execution time — method: {m}"} 
            ]
        ))

    fig.update_layout(
        title=f"{forecaster_names[0]} — Execution time `{methods[0]}`",
        xaxis=dict(
            tickmode="array",
            tickvals=list(version_to_num.values()),
            ticktext=list(version_to_num.keys()),
            title="skforecast version",
            tickangle=0,
            automargin=True,
        ),
        yaxis=dict(
            title="Execution time (s)", 
            automargin=True
        ),
        # template="plotly_white",
        margin=dict(l=70, r=20, t=100, b=60), 
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons_methods,
                showactive=True,
                direction="down",
                x=1.00,
                y=1.03,
                xanchor="right",
                yanchor="bottom",
                pad={"r": 2, "t": 0},
            ),
        ],
        legend=dict(title=""),
        showlegend=False,
    )

    fig.show()


def plot_benchmark_results_old(df, function_name, add_median=True, add_mean=True):
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
