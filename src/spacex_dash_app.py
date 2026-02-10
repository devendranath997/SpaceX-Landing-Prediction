"""
SpaceX Falcon 9 Landing Prediction - Interactive Plotly Dash Application
IBM Data Science Professional Certificate Capstone Project

This module implements an interactive web-based dashboard using Plotly Dash
for exploring and analyzing SpaceX Falcon 9 launch data. The dashboard provides
interactive visualizations including site selection, success rate analysis,
and payload impact analysis.

Features:
- Dropdown for launch site selection (All Sites, CCAFS SLC 40, VAFB SLC 4E, KSC LC 39A)
- Pie charts showing success/failure distribution by site
- Payload range slider for filtering
- Scatter plots with payload mass vs outcome
- Booster version color coding
- Real-time interactive callbacks

Author: Devendra Nath (devendranath997)
Date: 2024
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from datetime import datetime
from typing import Tuple, List


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = "../data/processed/spacex_launches_processed.csv"
LAUNCH_SITES = [
    'All Sites',
    'CCAFS SLC 40',
    'VAFB SLC 4E',
    'KSC LC 39A',
    'BOCA CHICA'
]

COLORS = {
    'success': '#2ca02c',    # Green
    'failure': '#d62728',    # Red
    'neutral': '#1f77b4'     # Blue
}


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading & Preparation
# ══════════════════════════════════════════════════════════════════════════════

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load processed data from CSV.

    Args:
        file_path: Path to CSV file

    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


def get_launch_site_column(df: pd.DataFrame) -> str:
    """
    Identify the launch site column name.

    Args:
        df: Input DataFrame

    Returns:
        Name of launch site column
    """
    for col in df.columns:
        if 'launchsite' in col.lower() or 'launch_site' in col.lower() or col == 'LaunchSite':
            return col
    return 'LaunchSite'


def get_payload_column(df: pd.DataFrame) -> str:
    """
    Identify the payload mass column name.

    Args:
        df: Input DataFrame

    Returns:
        Name of payload column
    """
    for col in df.columns:
        if 'payload' in col.lower() and 'mass' in col.lower():
            return col
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Callback Functions
# ══════════════════════════════════════════════════════════════════════════════

def create_pie_chart(df: pd.DataFrame, selected_site: str) -> go.Figure:
    """
    Create pie chart showing success/failure distribution.

    Args:
        df: Input DataFrame
        selected_site: Selected launch site

    Returns:
        Plotly Figure object
    """
    site_col = get_launch_site_column(df)

    # Filter data
    if selected_site != 'All Sites':
        filtered_df = df[df[site_col] == selected_site]
    else:
        filtered_df = df

    # Count outcomes
    if 'Class' in filtered_df.columns:
        outcome_counts = filtered_df['Class'].value_counts()
        labels = ['Failure', 'Success'] if 0 in outcome_counts.index else ['Success']
        values = [outcome_counts.get(0, 0), outcome_counts.get(1, 0)]
        colors_list = [COLORS['failure'], COLORS['success']]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors_list, line=dict(color='white', width=2)),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])

        title = f"Landing Success Rate - {selected_site}" if selected_site != 'All Sites' else "Landing Success Rate - All Sites"
        fig.update_layout(
            title=title,
            font=dict(size=12),
            height=400
        )

        return fig

    return go.Figure()


def create_scatter_plot(df: pd.DataFrame, payload_range: Tuple[float, float]) -> go.Figure:
    """
    Create scatter plot of payload mass vs landing outcome.

    Args:
        df: Input DataFrame
        payload_range: Tuple of (min_payload, max_payload)

    Returns:
        Plotly Figure object
    """
    payload_col = get_payload_column(df)

    if payload_col is None:
        return go.Figure()

    # Filter by payload range
    df_filtered = df[
        (df[payload_col] >= payload_range[0]) &
        (df[payload_col] <= payload_range[1])
    ].copy()

    # Identify booster version column
    booster_col = None
    for col in df.columns:
        if 'booster' in col.lower() or 'rocket' in col.lower():
            if col != 'Rocket Type':
                booster_col = col
                break

    # Create color mapping for Class
    color_map = {1: COLORS['success'], 0: COLORS['failure']}
    label_map = {1: 'Success', 0: 'Failure'}

    if booster_col and booster_col in df_filtered.columns:
        # Color by booster version
        fig = px.scatter(
            df_filtered,
            x='Flight Number' if 'Flight Number' in df_filtered.columns else range(len(df_filtered)),
            y=payload_col,
            color=booster_col,
            size='Flight Number' if 'Flight Number' in df_filtered.columns else None,
            hover_data={'Class': True, 'Flight Number': True},
            labels={
                'Flight Number': 'Flight Number',
                payload_col: 'Payload Mass (kg)',
                booster_col: 'Booster Version'
            }
        )
    else:
        # Color by outcome (Class)
        fig = go.Figure()

        for class_val in [0, 1]:
            mask = df_filtered['Class'] == class_val
            fig.add_trace(go.Scatter(
                x=df_filtered[mask].get('Flight Number', range(mask.sum())),
                y=df_filtered[mask][payload_col],
                mode='markers',
                name=label_map[class_val],
                marker=dict(
                    size=8,
                    color=color_map[class_val],
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>%{fullData.name}</b><br>Payload: %{y:.0f} kg<extra></extra>'
            ))

    fig.update_layout(
        title='Payload Mass vs Flight Number (Colored by Landing Outcome)',
        xaxis_title='Flight Number',
        yaxis_title='Payload Mass (kg)',
        height=400,
        hovermode='closest'
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Dash Application Setup
# ══════════════════════════════════════════════════════════════════════════════

def create_dash_app(df: pd.DataFrame) -> Dash:
    """
    Create and configure the Dash application.

    Args:
        df: Input DataFrame with launch data

    Returns:
        Configured Dash application
    """
    app = Dash(__name__)

    # Get payload column and range
    payload_col = get_payload_column(df)
    if payload_col:
        payload_min = df[payload_col].min()
        payload_max = df[payload_col].max()
    else:
        payload_min = 0
        payload_max = 10000

    # Define app layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1(
                "SpaceX Falcon 9 Landing Prediction Dashboard",
                style={'textAlign': 'center', 'color': '#1f77b4', 'marginBottom': 10}
            ),
            html.P(
                "IBM Data Science Professional Certificate Capstone Project",
                style={'textAlign': 'center', 'color': '#666', 'fontSize': 14}
            ),
        ], style={'padding': '20px', 'borderBottom': '2px solid #1f77b4'}),

        # Main content
        html.Div([
            # Control panel
            html.Div([
                html.Div([
                    html.Label("Select Launch Site:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='site-dropdown',
                        options=[{'label': site, 'value': site} for site in LAUNCH_SITES],
                        value='All Sites',
                        style={'width': '100%'}
                    ),
                ], style={'marginBottom': 20}),

                html.Div([
                    html.Label("Payload Mass Range (kg):", style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(
                        id='payload-slider',
                        min=payload_min,
                        max=payload_max,
                        value=[payload_min, payload_max],
                        marks={
                            int(payload_min): f'{int(payload_min)}',
                            int(payload_max/2): f'{int(payload_max/2)}',
                            int(payload_max): f'{int(payload_max)}'
                        },
                        tooltip={"placement": "bottom", "always_visible": True},
                        step=100
                    ),
                ], style={'marginBottom': 20}),

            ], style={
                'width': '100%',
                'padding': '20px',
                'backgroundColor': '#f5f5f5',
                'borderRadius': '5px',
                'marginBottom': '20px'
            }),

            # Visualizations
            html.Div([
                html.Div([
                    dcc.Graph(id='pie-chart')
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    dcc.Graph(id='scatter-plot')
                ], style={'width': '48%', 'display': 'inline-block'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),

            # Summary statistics
            html.Div(id='summary-stats', style={
                'padding': '20px',
                'backgroundColor': '#f9f9f9',
                'borderRadius': '5px',
                'marginTop': '20px',
                'borderLeft': '4px solid #1f77b4'
            }),

        ], style={'padding': '20px'}),

        # Footer
        html.Div([
            html.P(
                f"Data updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Total launches: {len(df)} | "
                f"Success rate: {(df['Class'].sum() / len(df) * 100):.1f}%",
                style={'textAlign': 'center', 'color': '#666', 'fontSize': 12, 'marginTop': 10}
            ),
        ], style={'padding': '20px', 'borderTop': '1px solid #ddd'}),

    ], style={
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#fff',
        'minHeight': '100vh'
    })

    # Define callbacks
    @app.callback(
        [Output('pie-chart', 'figure'),
         Output('scatter-plot', 'figure'),
         Output('summary-stats', 'children')],
        [Input('site-dropdown', 'value'),
         Input('payload-slider', 'value')]
    )
    def update_dashboard(selected_site: str, payload_range: list):
        """
        Update all dashboard components based on user selections.
        """
        site_col = get_launch_site_column(df)
        payload_col = get_payload_column(df)

        # Filter data
        if selected_site != 'All Sites':
            filtered_df = df[df[site_col] == selected_site].copy()
        else:
            filtered_df = df.copy()

        # Update pie chart
        pie_fig = create_pie_chart(df, selected_site)

        # Update scatter plot
        scatter_fig = create_scatter_plot(filtered_df, tuple(payload_range))

        # Calculate summary statistics
        total_launches = len(filtered_df)
        successful = (filtered_df['Class'] == 1).sum()
        success_rate = (successful / total_launches * 100) if total_launches > 0 else 0

        summary = html.Div([
            html.H3("Summary Statistics", style={'marginTop': 0}),
            html.Div([
                html.Div([
                    html.H4(f"{total_launches}", style={'color': '#1f77b4'}),
                    html.P("Total Launches")
                ], style={'textAlign': 'center', 'flex': 1}),
                html.Div([
                    html.H4(f"{successful}", style={'color': COLORS['success']}),
                    html.P("Successful Landings")
                ], style={'textAlign': 'center', 'flex': 1}),
                html.Div([
                    html.H4(f"{total_launches - successful}", style={'color': COLORS['failure']}),
                    html.P("Failed Landings")
                ], style={'textAlign': 'center', 'flex': 1}),
                html.Div([
                    html.H4(f"{success_rate:.1f}%", style={'color': '#ff7f0e'}),
                    html.P("Success Rate")
                ], style={'textAlign': 'center', 'flex': 1}),
            ], style={'display': 'flex', 'justifyContent': 'space-around'}),
        ])

        return pie_fig, scatter_fig, summary

    return app


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main execution function for Dash application.
    """
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "SpaceX Falcon 9 - Interactive Plotly Dash Dashboard".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    # Load data
    df = load_data(DATA_PATH)
    if df is None:
        print("✗ Failed to load data. Exiting.")
        return

    print(f"\n✓ Data loaded successfully: {df.shape[0]} launches")

    # Create Dash app
    print("✓ Creating Dash application...")
    app = create_dash_app(df)

    print("\n" + "="*80)
    print("DASH APPLICATION READY")
    print("="*80)
    print("✓ Dashboard created with following features:")
    print("  - Launch site selection dropdown")
    print("  - Payload mass range slider")
    print("  - Success/failure pie chart")
    print("  - Payload vs outcome scatter plot")
    print("  - Real-time summary statistics")
    print("\n✓ To run the application:")
    print("  - Execute: python spacex_dash_app.py")
    print("  - Open browser: http://127.0.0.1:8050/")
    print("  - Interact with dropdowns and sliders to explore data")

    # Run the app
    app.run_server(debug=True, host='127.0.0.1', port=8050)


if __name__ == "__main__":
    main()
