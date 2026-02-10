"""
SpaceX Falcon 9 Landing Prediction - Exploratory Data Analysis & Visualization
IBM Data Science Professional Certificate Capstone Project

This module performs comprehensive exploratory data analysis (EDA) and creates
publication-quality visualizations. It analyzes relationships between features
and the landing success outcome, revealing patterns and trends in SpaceX's
launch and landing data.

Key Visualizations:
- Scatter plots: Flight number vs payload mass (colored by landing outcome)
- Bar charts: Success rate by orbit type
- Line charts: Landing success trend over time and by year
- Distribution plots for key numerical features

Author: Devendra Nath (devendranath997)
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = "../data/processed/spacex_launches_processed.csv"
OUTPUT_DIR = "../images"
FIGURE_DPI = 300
FIGURE_SIZE_LARGE = (14, 8)
FIGURE_SIZE_MEDIUM = (12, 6)
FIGURE_SIZE_SMALL = (10, 5)

# Set style for all plots
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load processed data from CSV file.

    Args:
        file_path: Path to input CSV

    Returns:
        Loaded DataFrame or None if error occurs
    """
    print("\n" + "="*80)
    print("LOADING DATA FOR EDA")
    print("="*80)

    try:
        df = pd.read_csv(file_path)
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Summary Statistics
# ══════════════════════════════════════════════════════════════════════════════

def print_summary_statistics(df: pd.DataFrame) -> None:
    """
    Print comprehensive summary statistics for the dataset.

    Args:
        df: DataFrame to analyze
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    print(f"\nData Types:")
    print(df.dtypes)

    print(f"\nNumerical Summary:")
    print(df.describe())

    if 'Class' in df.columns:
        print(f"\nTarget Variable Distribution:")
        print(df['Class'].value_counts().sort_index())
        print(f"Success Rate: {(df['Class'] == 1).sum() / len(df) * 100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Visualization Functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_flight_number_vs_payload(df: pd.DataFrame) -> None:
    """
    Create scatter plot of flight number vs payload mass,
    colored by landing outcome (Class).

    Args:
        df: DataFrame containing flight data
    """
    print("\nGenerating: Flight Number vs Payload Mass scatter plot...")

    # Find payload-related columns
    payload_col = None
    for col in df.columns:
        if 'payload' in col.lower() and 'mass' in col.lower():
            payload_col = col
            break

    if payload_col is None:
        print("  ⚠ Payload mass column not found. Skipping.")
        return

    if 'Flight Number' not in df.columns and 'flight_number' not in df.columns:
        print("  ⚠ Flight Number column not found. Skipping.")
        return

    flight_col = 'Flight Number' if 'Flight Number' in df.columns else 'flight_number'

    # Create plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE, dpi=FIGURE_DPI)

    # Prepare data
    df_plot = df.dropna(subset=[flight_col, payload_col, 'Class'])

    # Scatter plot with color coding
    colors = {0: '#d62728', 1: '#2ca02c'}  # Red for failure, Green for success
    labels = {0: 'Failure', 1: 'Success'}

    for class_val in [0, 1]:
        mask = df_plot['Class'] == class_val
        ax.scatter(df_plot[mask][flight_col], df_plot[mask][payload_col],
                  c=colors[class_val], label=labels[class_val], alpha=0.6, s=100, edgecolors='black')

    ax.set_xlabel('Flight Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Payload Mass (kg)', fontsize=12, fontweight='bold')
    ax.set_title('SpaceX Falcon 9: Flight Number vs Payload Mass\n(Colored by Landing Outcome)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, title='Landing Outcome')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/01_flight_payload_scatter.png"
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    plt.close()


def plot_success_rate_by_orbit(df: pd.DataFrame) -> None:
    """
    Create bar chart showing success rate by orbit type.

    Args:
        df: DataFrame containing orbit and outcome data
    """
    print("\nGenerating: Success Rate by Orbit Type bar chart...")

    # Find orbit column
    orbit_col = None
    for col in df.columns:
        if 'orbit' in col.lower():
            orbit_col = col
            break

    if orbit_col is None:
        print("  ⚠ Orbit column not found. Skipping.")
        return

    # Calculate success rate by orbit
    if 'Class' in df.columns:
        success_by_orbit = df.groupby(orbit_col)['Class'].agg(['sum', 'count'])
        success_by_orbit['success_rate'] = (success_by_orbit['sum'] / success_by_orbit['count'] * 100)
        success_by_orbit = success_by_orbit.sort_values('success_rate', ascending=False)

        # Create plot
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_MEDIUM, dpi=FIGURE_DPI)

        bars = ax.bar(range(len(success_by_orbit)), success_by_orbit['success_rate'],
                     color=plt.cm.RdYlGn(success_by_orbit['success_rate'] / 100))

        ax.set_xticks(range(len(success_by_orbit)))
        ax.set_xticklabels(success_by_orbit.index, rotation=45, ha='right')
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Orbit Type', fontsize=12, fontweight='bold')
        ax.set_title('SpaceX Falcon 9: Landing Success Rate by Orbit Type',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 105)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%\n(n={int(success_by_orbit["count"].iloc[i])})',
                   ha='center', va='bottom', fontsize=9)

        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        output_path = f"{OUTPUT_DIR}/02_success_by_orbit.png"
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_path}")
        plt.close()
    else:
        print("  ⚠ Class column not found. Skipping.")


def plot_success_trend_over_time(df: pd.DataFrame) -> None:
    """
    Create line chart showing landing success trend over time/flight number.

    Args:
        df: DataFrame containing flight and outcome data
    """
    print("\nGenerating: Success Trend Over Time line chart...")

    flight_col = None
    for col in df.columns:
        if 'flight' in col.lower() and 'number' in col.lower():
            flight_col = col
            break

    if flight_col is None:
        print("  ⚠ Flight number column not found. Skipping.")
        return

    if 'Class' not in df.columns:
        print("  ⚠ Class column not found. Skipping.")
        return

    # Sort by flight number and calculate cumulative success rate
    df_sorted = df.sort_values(flight_col).copy()
    df_sorted['cumulative_success'] = df_sorted['Class'].cumsum()
    df_sorted['cumulative_total'] = range(1, len(df_sorted) + 1)
    df_sorted['rolling_success_rate'] = (df_sorted['cumulative_success'] / df_sorted['cumulative_total'] * 100)

    # Calculate rolling mean (window of 5)
    df_sorted['rolling_mean'] = df_sorted['Class'].rolling(window=5, center=True).mean() * 100

    # Create plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE, dpi=FIGURE_DPI)

    ax.plot(df_sorted[flight_col], df_sorted['rolling_success_rate'],
           linewidth=2, color='#1f77b4', label='Cumulative Success Rate', marker='o', markersize=3)
    ax.plot(df_sorted[flight_col], df_sorted['rolling_mean'],
           linewidth=2.5, color='#ff7f0e', label='5-Launch Rolling Average', linestyle='--')

    ax.set_xlabel('Flight Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('SpaceX Falcon 9: Landing Success Trend Over Time',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/03_success_trend.png"
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    plt.close()


def plot_success_by_year(df: pd.DataFrame) -> None:
    """
    Create bar chart showing success rate by year.

    Args:
        df: DataFrame with year and outcome data
    """
    print("\nGenerating: Success Rate by Year bar chart...")

    # Create year column if not exists
    if 'Year' not in df.columns:
        # Try to extract from date columns
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if len(date_cols) > 0:
            df['temp_date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df['Year'] = df['temp_date'].dt.year
        else:
            print("  ⚠ Cannot determine year. Skipping.")
            return

    if 'Class' not in df.columns:
        print("  ⚠ Class column not found. Skipping.")
        return

    # Calculate success rate by year
    success_by_year = df.groupby('Year')['Class'].agg(['sum', 'count'])
    success_by_year['success_rate'] = (success_by_year['sum'] / success_by_year['count'] * 100)
    success_by_year = success_by_year.sort_index()

    # Create plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_MEDIUM, dpi=FIGURE_DPI)

    bars = ax.bar(success_by_year.index, success_by_year['success_rate'],
                 color=plt.cm.viridis(np.linspace(0.3, 0.9, len(success_by_year))),
                 width=0.6, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('SpaceX Falcon 9: Landing Success Rate by Year',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/04_success_by_year.png"
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    plt.close()


def plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Create pie chart showing overall Class distribution.

    Args:
        df: DataFrame with Class information
    """
    print("\nGenerating: Class Distribution pie chart...")

    if 'Class' not in df.columns:
        print("  ⚠ Class column not found. Skipping.")
        return

    # Calculate distribution
    class_counts = df['Class'].value_counts().sort_index()
    class_labels = {0: 'Failure', 1: 'Success'}
    labels = [class_labels.get(i, f'Class {i}') for i in class_counts.index]

    # Create plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL, dpi=FIGURE_DPI)

    colors = ['#d62728', '#2ca02c']  # Red, Green
    wedges, texts, autotexts = ax.pie(class_counts, labels=labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90, textprops={'fontsize': 11})

    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    ax.set_title('SpaceX Falcon 9: Landing Outcome Distribution',
                fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/05_class_distribution.png"
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main execution function for exploratory data analysis and visualization.
    """
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "SpaceX Falcon 9 - Exploratory Data Analysis & Visualization".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    start_time = datetime.now()
    print(f"\nExecution started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data(DATA_PATH)
    if df is None:
        print("✗ Failed to load data. Exiting.")
        return

    # Print summary statistics
    print_summary_statistics(df)

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_flight_number_vs_payload(df)
    plot_success_rate_by_orbit(df)
    plot_success_trend_over_time(df)
    plot_success_by_year(df)
    plot_class_distribution(df)

    # Completion summary
    print("\n" + "="*80)
    print("EDA ANALYSIS SUMMARY")
    print("="*80)
    print("✓ All visualizations generated successfully")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("\nGenerated Visualizations:")
    print("  1. Flight Number vs Payload Mass Scatter Plot")
    print("  2. Success Rate by Orbit Type Bar Chart")
    print("  3. Landing Success Trend Over Time Line Chart")
    print("  4. Success Rate by Year Bar Chart")
    print("  5. Class Distribution Pie Chart")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n✓ Execution completed in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
