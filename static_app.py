import os
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
from scipy.signal import find_peaks
from helpers.plot import plot_time_series, maintain_zoom
from pipeline.data_io import read_device_csv
from pipeline.preprocess import filter_signal, segment_signal
from pipeline.alignment import align_segments
from pipeline.rf_model import predict_rf_model

# CONSTANTS
Fs = 500
FFT_skip = 40

# Load raw data
filepath = "data/raw/center-test.csv"
df = read_device_csv(filepath, logging=False)
df = df.with_columns(pl.Series("time", np.arange(df.height) / Fs))
df = df.with_columns(pl.Series("z", df["z"] / 262144))

t = df["time"].to_numpy()
z = df["z"].to_numpy()

app = Dash(name=__name__)
app.layout = html.Div([
    html.Div([
        html.H1(f"SCG Signal Visualization - File: {filepath}"),
        dcc.Graph(id="raw-signal", figure=plot_time_series(t, z, title="Raw Signal")),
    ], style={"textAlign": "center"}),

    # Bandpass filter
    html.Div([
        html.H2("Bandpass Filtering"),
        html.Label("Low cutoff (Hz): "),
        dcc.Input(id="lowcut", type="number", min=0.1, max=100, step=0.1, value=1, debounce=True),
        html.Label("High cutoff (Hz): "),
        dcc.Input(id="highcut", type="number", min=10, max=200, step=1, value=150, debounce=True),
        html.Label("Filter order: "),
        dcc.Input(id="order", type="number", min=1, max=6, step=1, value=2, debounce=True),
        dcc.Graph(id="filtered-signal"),
    ], style={"margin": "10px"}),

    # FFT
    html.Div([
        html.Div([
            dcc.Graph(id="raw-fft", figure=px.line(
                x=np.fft.rfftfreq(len(z), 1/Fs)[FFT_skip:],
                y=np.abs(np.fft.rfft(z - np.mean(z)))[FFT_skip:],
                labels={"x": "Frequency (Hz)", "y": "Magnitude"},
                title="FFT of Raw Signal"
            ))
        ], style={"flex": 1, "margin": 5}),
        html.Div([dcc.Graph(id="filtered-fft")], style={"flex": 1, "margin": 5}),
    ], style={"display": "flex"}),

    # Downward peaks
    html.Div([
        html.H2("Downward Peak Detection"),
        html.Label("Distance (samples): "),
        dcc.Input(id="distance", type="number", min=10, max=200, step=1, value=int(Fs/3), debounce=True),
        html.Label("Prominence: "),
        dcc.Input(id="prominence", type="number", min=0, max=1, step=0.001, value=0.012, debounce=True),
        html.Label("Tolerance (std dev): "),
        dcc.Input(id="tolerance", type="number", min=0, max=10, step=0.1, value=1.5, debounce=True),
        dcc.Graph(id="downpeaks"),
    ], style={"margin": 10}),

    # Segmentation
    html.Div([
        html.H2("Segmentation"),
        html.Label("Segment width (samples): "),
        dcc.Input(id="segment-width", type="number", min=50, max=250, step=1, value=150, debounce=True),
        html.Label("Averaging window: "),
        dcc.Input(id="averaging-window", type="number", min=0, max=20, step=1, value=10, debounce=True),
        dcc.Graph(id="segmentation"),
    ], style={"margin": 10}),

    # Alignment Section
    html.Div([
        html.H2("Alignment"),
        html.Div([
            html.Label("Reference CSV paths (comma-separated): "),
            dcc.Input(
                id="reference-paths",
                type="text",
                value=",".join([f"data/references/center{i}_reference.csv" for i in range(1, 6)] + [f"data/references/left{i}_reference.csv" for i in range(1, 6)]),
                style={"width": "80%"},
                debounce=True
            ),
        ], style={"margin": "10px"}),
        html.Div([
            html.Button('Run Alignment', id='align-button', n_clicks=0,
                       style={'fontSize': '18px', 'padding': '10px 25px', 'margin': '10px'})
        ], style={'textAlign': 'center'}),
        
        # Alignment statistics
        html.Div(id='alignment-stats', style={
            'backgroundColor': '#f0f0f0',
            'padding': '15px',
            'margin': '20px',
            'borderRadius': '5px',
            'fontFamily': 'monospace'
        }),
        
        # Alignment visualization
        html.Div([
            html.Label("Aligned Segment: "),
            dcc.Slider(id='aligned-seg-slider', min=0, max=0, step=1, value=0,
                      marks={}, tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'margin': '20px'}),
        dcc.Graph(id="aligned-segment"),
        
        # Reference selection for comparison
        html.Div([
            html.Label("Reference for comparison: "),
            dcc.Slider(id='aligned-ref-slider', min=0, max=0, step=1, value=0,
                      marks={}, tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'margin': '20px'}),
        
    ], id='alignment-section', style={'margin': 10, 'display': 'none'}),

    # Prediction Section
    html.Div([
        html.H2("Model Prediction"),
        html.Div([
            html.Label("Model path: "),
            dcc.Input(
                id="model-path",
                type="text",
                value="models/sensor_classifier_v1.pkl",
                style={"width": "60%"},
                debounce=True
            ),
        ], style={"margin": "10px"}),
        html.Div([
            html.Button('Run Prediction', id='predict-button', n_clicks=0,
                       style={'fontSize': '18px', 'padding': '10px 25px', 'margin': '10px'})
        ], style={'textAlign': 'center'}),
        
        # Prediction results
        html.Div(id='prediction-results', style={
            'backgroundColor': '#f0f0f0',
            'padding': '15px',
            'margin': '20px',
            'borderRadius': '5px',
            'fontFamily': 'monospace'
        }),
        
    ], id='prediction-section', style={'margin': 10, 'display': 'none'}),

    # Stores
    dcc.Store(id="filtered-data"),
    dcc.Store(id="downpeaks-data"),
    dcc.Store(id="filtered-downpeaks-data"),
    dcc.Store(id="segmented-data"),
    dcc.Store(id="aligned-data"),
    dcc.Store(id="references-data"),
    dcc.Store(id="segments-temp-path"),
])

# -------------------------
# Callbacks
# -------------------------

@app.callback(
    [Output("filtered-signal", "figure"),
     Output("filtered-data", "data")],
    [Input("lowcut", "value"),
     Input("highcut", "value"),
     Input("order", "value"),
     Input("filtered-signal", "relayoutData")]
)
def update_filtered_signal(lowcut, highcut, order, relayout_data):
    if not lowcut or not highcut or lowcut >= highcut or highcut >= Fs/2:
        return plot_time_series(t, z, title="Invalid filter range"), None

    df_filt, _ = filter_signal(
        filepath=filepath, 
        fs=Fs, 
        lowcut=lowcut, 
        highcut=highcut, 
        order=order,
        save=False
    )
    z_filt = df_filt["z"].to_numpy()
    
    fig = plot_time_series(t, z_filt, title=f"Bandpass ({lowcut}-{highcut} Hz, order {order})")
    maintain_zoom(relayout_data, fig)
    return fig, z_filt.tolist()


@app.callback(
    Output("filtered-fft", "figure"),
    [Input("filtered-data", "data"),
     Input("filtered-fft", "relayoutData")]
)
def update_filtered_fft(z_filt_data, relayout_data):
    if z_filt_data is None:
        return px.line(title="No filtered data available")
    z_filt = np.array(z_filt_data)
    fig = px.line(
        x=np.fft.rfftfreq(len(z_filt), 1/Fs)[FFT_skip:],
        y=np.abs(np.fft.rfft(z_filt - np.mean(z_filt)))[FFT_skip:],
        labels={"x": "Frequency (Hz)", "y": "Magnitude"},
        title="FFT of Filtered Signal"
    )
    maintain_zoom(relayout_data, fig)
    return fig


@app.callback(
    [Output("downpeaks", "figure"),
     Output("downpeaks-data", "data"),
     Output("filtered-downpeaks-data", "data")],
    [Input("filtered-data", "data"),
     Input("distance", "value"),
     Input("prominence", "value"),
     Input("tolerance", "value"),
     Input("downpeaks", "relayoutData")]
)
def update_downpeaks(z_filt_data, distance, prominence, tolerance, relayout_data):
    if z_filt_data is None:
        return px.line(title="No filtered data available"), [], []

    z_filt = np.array(z_filt_data)
    downpeaks, _ = find_peaks(-z_filt, distance=int(distance), prominence=float(prominence))

    if len(downpeaks) == 0:
        return plot_time_series(t, z_filt, title="No down-peaks"), [], []

    # Outlier removal
    vals = z_filt[downpeaks]
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lower_cut = q1 - iqr * float(tolerance)
    upper_cut = q3 + iqr * float(tolerance)
    filtered = [i for i in downpeaks if lower_cut <= z_filt[i] <= upper_cut]

    fig = plot_time_series(t, z_filt, title="Detected Down-peaks")
    fig.add_scatter(x=t[downpeaks], y=z_filt[downpeaks], mode="markers", marker=dict(color="green", size=8), name="Down-peaks")
    outliers = [i for i in downpeaks if i not in filtered]
    if outliers:
        fig.add_scatter(x=t[outliers], y=z_filt[outliers], mode="markers", marker=dict(color="red", size=10), name="Outliers")

    fig.add_hline(y=lower_cut, line=dict(color='red', dash='dot'), annotation_text="Lower cutoff", annotation_position="bottom right")
    fig.add_hline(y=upper_cut, line=dict(color='red', dash='dot'), annotation_text="Upper cutoff", annotation_position="top right")

    maintain_zoom(relayout_data, fig)
    return fig, downpeaks.tolist(), filtered if filtered else downpeaks.tolist()


@app.callback(
    [Output("segmentation", "figure"),
     Output("segmented-data", "data"),
     Output("segments-temp-path", "data"),
     Output("alignment-section", "style")],
    [Input("filtered-data", "data"),
     Input("filtered-downpeaks-data", "data"),
     Input("averaging-window", "value"),
     Input("segment-width", "value"),
     Input("segmentation", "relayoutData"),
     Input("distance", "value"),
     Input("prominence", "value"),
     Input("tolerance", "value")]
)
def update_segmentation(z_filt_data, filtered_downpeaks_data, averaging_window, user_segment_width, relayout_data, distance, prominence, tolerance):
    if z_filt_data is None or not filtered_downpeaks_data:
        return px.line(title="No data available"), None, None, {'margin': 10, 'display': 'none'}

    df_tmp = pl.DataFrame({"z": z_filt_data})

    segments, segments_path = segment_signal(
        df=df_tmp,
        filepath=filepath,
        fs=Fs,
        distance=int(distance),
        prominence=float(prominence),
        tolerance=float(tolerance),
        segment_width=int(user_segment_width),
        averaging_window=int(averaging_window),
        save=True,
        output_dir="data/temp"
    )

    if len(segments) == 0 or segments_path is None:
        return px.line(title="No valid segments"), None, None, {'margin': 10, 'display': 'none'}

    fig = go.Figure()
    z_matrix = np.stack(segments)
    avg_z = np.mean(z_matrix, axis=0)
    avg_time = np.arange(len(avg_z))/Fs

    # Plot individual segments
    for seg in segments:
        seg_time = np.arange(len(seg))/Fs
        fig.add_trace(go.Scatter(x=seg_time, y=seg, mode='lines', line=dict(color='blue', width=1), showlegend=False))

    avg_dps_idx, _ = find_peaks(-avg_z)
    avg_ups_idx, _ = find_peaks(avg_z)

    # Overlay average segment
    fig.add_trace(go.Scatter(x=avg_time, y=avg_z, mode='lines', line=dict(color='red', width=4), name='Average Segment'))
    fig.add_scatter(x=avg_time[avg_dps_idx], y=avg_z[avg_dps_idx], mode="markers", marker=dict(color="green", size=8), name="Down-peaks")
    fig.add_scatter(x=avg_time[avg_ups_idx], y=avg_z[avg_ups_idx], mode="markers", marker=dict(color="blue", size=8), name="Up-peaks")

    maintain_zoom(relayout_data, fig)

    # Show alignment section now that we have segments
    return fig, {'segments': [seg.tolist() for seg in segments]}, segments_path, {'margin': 10, 'display': 'block'}


@app.callback(
    [Output("aligned-data", "data"),
     Output("references-data", "data"),
     Output("aligned-seg-slider", "max"),
     Output("aligned-seg-slider", "marks"),
     Output("aligned-ref-slider", "max"),
     Output("aligned-ref-slider", "marks"),
     Output("alignment-stats", "children"),
     Output("prediction-section", "style")],
    [Input("align-button", "n_clicks")],
    [State("segments-temp-path", "data"),
     State("reference-paths", "value")]
)
def run_alignment(n_clicks, segments_path, reference_paths_str):
    if n_clicks == 0 or segments_path is None:
        return None, None, 0, {}, 0, {}, "", {'margin': 10, 'display': 'none'}
    
    # Parse reference paths
    ref_paths = [p.strip() for p in reference_paths_str.split(',')]
    
    # Load references for display
    try:
        references = [pl.read_csv(path) for path in ref_paths]
    except Exception as e:
        return None, None, 0, {}, 0, {}, html.Div(f"Error loading references: {str(e)}", style={'color': 'red'}), {'margin': 10, 'display': 'none'}
    
    # Run alignment using pipeline module
    try:
        aligned_results, _ = align_segments(
            segments_path=segments_path,
            reference_paths=ref_paths,
            fs=Fs,
            save=False
        )
    except Exception as e:
        return None, None, 0, {}, 0, {}, html.Div(f"Error during alignment: {str(e)}", style={'color': 'red'}), {'margin': 10, 'display': 'none'}
    
    if len(aligned_results) == 0:
        return None, None, 0, {}, 0, {}, html.Div("No segments aligned successfully", style={'color': 'orange'}), {'margin': 10, 'display': 'none'}
    
    # Calculate statistics
    scores = np.array([score for _, _, _, score, _ in aligned_results])
    
    stats = {
        "min": np.min(scores),
        "max": np.max(scores),
        "mean": np.mean(scores),
        "median": np.median(scores),
        "q1": np.percentile(scores, 25),
        "q3": np.percentile(scores, 75),
        "std": np.std(scores),
        "count": len(scores)
    }
    
    stats_text = html.Div([
        html.H4("Alignment Score Statistics:", style={'marginBottom': '10px'}),
        html.Pre('\n'.join([
            f"  {k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"  {k}: {v}"
            for k, v in stats.items()
        ]))
    ])
    
    num_aligned = len(aligned_results)
    num_refs = len(references)
    
    seg_marks = {i: str(i) for i in range(num_aligned)}
    ref_marks = {i: str(i) for i in range(num_refs)}
    
    # Store aligned data and references
    aligned_data = {
        'results': [
            {
                'time': time.tolist(),
                'amplitude': amp.tolist(),
                'features': feat.tolist(),
                'score': float(score),
                'ref_idx': int(ref_idx)
            }
            for time, amp, feat, score, ref_idx in aligned_results
        ]
    }
    
    references_data = {
        'references': [
            {
                'time': ref['Time'].to_numpy().tolist(),
                'amplitude': ref['Amplitude'].to_numpy().tolist(),
                'feature': ref['Feature'].to_numpy().tolist()
            }
            for ref in references
        ]
    }
    
    # Show prediction section after successful alignment
    return aligned_data, references_data, num_aligned - 1, seg_marks, num_refs - 1, ref_marks, stats_text, {'margin': 10, 'display': 'block'}


@app.callback(
    [Output("aligned-segment", "figure"),
     Output("aligned-ref-slider", "value")], 
    [Input("aligned-seg-slider", "value"),
     Input("aligned-ref-slider", "value")],
    [State("aligned-data", "data"),
     State("references-data", "data")]
)
def update_aligned_plot(seg_idx, ref_idx, aligned_data, references_data):
    if aligned_data is None or references_data is None:
        return go.Figure(), 0
    
    # Get aligned segment
    seg_data = aligned_data['results'][seg_idx]
    best_ref_idx = seg_data['ref_idx']
    time = np.array(seg_data['time'])
    amplitude = np.array(seg_data['amplitude'])
    features = np.array(seg_data['features'])
    score = seg_data['score']
    
    # If segment slider changed, reset to best reference
    if callback_context.triggered and 'aligned-seg-slider' in callback_context.triggered[0]['prop_id']:
        ref_idx = best_ref_idx
    
    # Get reference
    ref_data = references_data['references'][ref_idx]
    ref_time = np.array(ref_data['time'])
    ref_amplitude = np.array(ref_data['amplitude'])
    ref_feature = np.array(ref_data['feature'])
    
    # Get feature points
    feature_mask = features >= 0
    seg_feat_time = time[feature_mask]
    seg_feat_amp = amplitude[feature_mask]
    seg_feat_labels = features[feature_mask]
    
    ref_feat_mask = ref_feature >= 0
    ref_feat_time = ref_time[ref_feat_mask]
    ref_feat_amp = ref_amplitude[ref_feat_mask]
    ref_feat_labels = ref_feature[ref_feat_mask]
    
    # Create figure
    fig = go.Figure()
    
    # Segment signal
    fig.add_trace(go.Scatter(
        x=time, y=amplitude,
        mode='lines',
        name=f'Segment {seg_idx}',
        line=dict(color='blue', width=2)
    ))
    
    # Segment features
    fig.add_trace(go.Scatter(
        x=seg_feat_time, y=seg_feat_amp,
        mode='markers+text',
        name='Segment Features',
        marker=dict(color='green', size=10),
        text=seg_feat_labels.astype(int),
        textposition='top center',
        textfont=dict(size=10, color='green')
    ))
    
    # Reference overlay
    fig.add_trace(go.Scatter(
        x=ref_time, y=ref_amplitude,
        mode='lines',
        name=f'Reference {ref_idx}',
        line=dict(color='red', width=1, dash='dash'),
        opacity=0.5
    ))
    
    # Reference features
    fig.add_trace(go.Scatter(
        x=ref_feat_time, y=ref_feat_amp,
        mode='markers+text',
        name='Reference Features',
        marker=dict(color='red', size=6),
        text=ref_feat_labels.astype(int),
        textposition='bottom center',
        textfont=dict(size=9, color='red'),
        opacity=0.5
    ))

    # Calculate y-axis bounds from current data
    all_amplitudes = np.concatenate([amplitude, ref_amplitude])
    y_min = np.min(all_amplitudes) * 1.1
    y_max = np.max(all_amplitudes) * 1.1
    
    title = f'Aligned Segment {seg_idx} (Score: {score:.2f}, Best Match: Ref {best_ref_idx}, Viewing: Ref {ref_idx})'
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        yaxis=dict(range=[y_min, y_max]),
        height=500,
        showlegend=True
    )
    
    return fig, ref_idx


@app.callback(
    Output("prediction-results", "children"),
    [Input("predict-button", "n_clicks")],
    [State("aligned-data", "data"),
     State("model-path", "value")]
)
def run_prediction(n_clicks, aligned_data, model_path):
    if n_clicks == 0 or aligned_data is None:
        return ""
    
    # Convert aligned_data back to list of tuples format
    try:
        aligned_results = [
            (
                np.array(seg['time']),
                np.array(seg['amplitude']),
                np.array(seg['features']),
                seg['score'],
                seg['ref_idx']
            )
            for seg in aligned_data['results']
        ]
    except Exception as e:
        return html.Div(f"Error preparing data: {str(e)}", style={'color': 'red'})
    
    # Run prediction
    try:
        most_common, all_preds, probs = predict_rf_model(
            model_path=model_path,
            aligned_data=aligned_results,
            score_percentile_cutoff=75.0,
            return_probabilities=False
        )
    except Exception as e:
        return html.Div(f"Error during prediction: {str(e)}", style={'color': 'red'})
    
    if most_common is None:
        return html.Div("Prediction failed", style={'color': 'red'})
    
    # Load model to get class names
    import joblib
    try:
        model_data = joblib.load(model_path)
        class_names = model_data['class_names']
    except:
        class_names = [f"Class {i}" for i in range(max(all_preds) + 1)]
    
    # Format results
    result_text = html.Div([
        html.H4("Prediction Results:", style={'marginBottom': '10px'}),
        html.Div([
            html.P(f"Predicted Class: {class_names[most_common]}", 
                   style={'fontSize': '18px', 'fontWeight': 'bold', 'color': 'green'}),
            html.Hr(),
            html.P("Per-Segment Breakdown:"),
            html.Pre('\n'.join([
                f"  {class_names[i]}: {np.sum(all_preds == i)} segments ({np.sum(all_preds == i)/len(all_preds)*100:.1f}%)"
                for i in range(len(class_names))
            ]))
        ])
    ])
    
    return result_text


if __name__ == "__main__":
    app.run(debug=True)
