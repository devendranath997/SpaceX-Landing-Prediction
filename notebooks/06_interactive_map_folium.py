"""
SpaceX Falcon 9 Landing Prediction - Interactive Launch Site Map
IBM Data Science Professional Certificate Capstone Project

This module creates an interactive web map using Folium that visualizes SpaceX
launch sites and their landing success rates. The map includes markers for each
site, popup information, and color-coded indicators for success/failure outcomes.

Key Features:
- Create Folium map centered on SpaceX launch sites
- Add markers with site information (name, location, coordinates)
- Color-coded markers: Green (success), Red (failure)
- Circle markers showing site coverage area
- Popups with detailed site statistics
- Save as interactive HTML file for browser viewing

Author: Devendra Nath (devendranath997)
Date: 2024
"""

import pandas as pd
import folium
from folium import plugins
from datetime import datetime
from typing import Optional, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = "../data/processed/spacex_launches_processed.csv"
OUTPUT_PATH = "../interactive/spacex_launch_sites_map.html"

# Known SpaceX launch site coordinates (latitude, longitude)
LAUNCH_SITES_COORDS = {
    'CCAFS SLC 40': (28.5621, -80.5771),
    'VAFB SLC 4E': (34.7331, -120.5271),
    'KSC LC 39A': (28.6239, -80.6042),
    'Omelek': (7.2906, 171.7845),  # Omelek Island
    'BOCA CHICA': (25.9973, -97.1569),
}

# Color scheme for markers
COLOR_SUCCESS = '#2ca02c'  # Green
COLOR_FAILURE = '#d62728'  # Red
COLOR_NEUTRAL = '#1f77b4'  # Blue


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading & Preparation
# ══════════════════════════════════════════════════════════════════════════════

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load processed launch data from CSV.

    Args:
        file_path: Path to CSV file

    Returns:
        Loaded DataFrame or None if error occurs
    """
    print("\n" + "="*80)
    print("LOADING DATA FOR MAP VISUALIZATION")
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


def prepare_site_statistics(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate statistics for each launch site.

    Args:
        df: DataFrame with launch data

    Returns:
        Dictionary with site statistics
    """
    print("\n" + "="*80)
    print("PREPARING LAUNCH SITE STATISTICS")
    print("="*80)

    # Find launch site column
    site_col = None
    for col in df.columns:
        if 'launchsite' in col.lower() or 'launch_site' in col.lower() or col == 'LaunchSite':
            site_col = col
            break

    if site_col is None:
        print("✗ Launch site column not found")
        return {}

    site_stats = {}

    for site in df[site_col].unique():
        if pd.isna(site):
            continue

        site_data = df[df[site_col] == site]

        if 'Class' in df.columns:
            successes = (site_data['Class'] == 1).sum()
            total = len(site_data)
            success_rate = (successes / total * 100) if total > 0 else 0
        else:
            successes = 0
            total = len(site_data)
            success_rate = 0

        site_stats[site] = {
            'total_launches': total,
            'successful_landings': successes,
            'failed_landings': total - successes,
            'success_rate': success_rate
        }

        print(f"✓ {site}")
        print(f"    Total Launches: {total}")
        print(f"    Successful: {successes} ({success_rate:.1f}%)")
        print(f"    Failed: {total - successes}")

    return site_stats


# ══════════════════════════════════════════════════════════════════════════════
# Map Creation Functions
# ══════════════════════════════════════════════════════════════════════════════

def create_base_map() -> folium.Map:
    """
    Create base Folium map centered on SpaceX facilities.

    Returns:
        Folium Map object
    """
    print("\n" + "="*80)
    print("CREATING BASE MAP")
    print("="*80)

    # Center map on continental US (weighted towards coastal areas)
    center_lat = 28.5
    center_lon = -95.0

    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles='OpenStreetMap'
    )

    print("✓ Base map created")
    print(f"  Center: ({center_lat}, {center_lon})")
    print(f"  Zoom: 4")

    return map_object


def add_site_markers(map_object: folium.Map, site_stats: Dict[str, Dict]) -> None:
    """
    Add markers to map for each launch site with popup information.

    Args:
        map_object: Folium Map object
        site_stats: Dictionary with site statistics
    """
    print("\n" + "="*80)
    print("ADDING LAUNCH SITE MARKERS")
    print("="*80)

    for site_name, stats in site_stats.items():
        # Get coordinates
        if site_name not in LAUNCH_SITES_COORDS:
            print(f"⚠ Coordinates not found for {site_name}. Skipping.")
            continue

        lat, lon = LAUNCH_SITES_COORDS[site_name]

        # Determine marker color based on success rate
        if stats['success_rate'] >= 66:
            marker_color = COLOR_SUCCESS
            icon_color = 'green'
        elif stats['success_rate'] >= 33:
            marker_color = COLOR_NEUTRAL
            icon_color = 'blue'
        else:
            marker_color = COLOR_FAILURE
            icon_color = 'red'

        # Create popup text
        popup_text = f"""
        <b>{site_name}</b><br>
        <hr style="margin: 5px 0;">
        Total Launches: {stats['total_launches']}<br>
        Successful: {stats['successful_landings']} ({stats['success_rate']:.1f}%)<br>
        Failed: {stats['failed_landings']}<br>
        <br>
        <small>Coordinates: ({lat:.4f}, {lon:.4f})</small>
        """

        # Add marker
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{site_name} - {stats['success_rate']:.1f}% Success",
            icon=folium.Icon(color=icon_color, icon='info-sign')
        ).add_to(map_object)

        print(f"✓ Marker added: {site_name}")
        print(f"    Location: ({lat}, {lon})")
        print(f"    Success Rate: {stats['success_rate']:.1f}%")


def add_site_circles(map_object: folium.Map, site_stats: Dict[str, Dict]) -> None:
    """
    Add circle markers showing launch site coverage areas.

    Args:
        map_object: Folium Map object
        site_stats: Dictionary with site statistics
    """
    print("\n" + "="*80)
    print("ADDING LAUNCH SITE COVERAGE CIRCLES")
    print("="*80)

    for site_name, stats in site_stats.items():
        if site_name not in LAUNCH_SITES_COORDS:
            continue

        lat, lon = LAUNCH_SITES_COORDS[site_name]

        # Circle radius based on number of launches (in kilometers, scaled to map)
        radius = min(stats['total_launches'] * 5000, 50000)

        # Circle color based on success rate
        if stats['success_rate'] >= 66:
            color = COLOR_SUCCESS
            fill_color = COLOR_SUCCESS
        elif stats['success_rate'] >= 33:
            color = COLOR_NEUTRAL
            fill_color = COLOR_NEUTRAL
        else:
            color = COLOR_FAILURE
            fill_color = COLOR_FAILURE

        # Add circle
        folium.Circle(
            location=[lat, lon],
            radius=radius,
            popup=f"{site_name}: {stats['success_rate']:.1f}%",
            color=color,
            fill=True,
            fillColor=fill_color,
            fillOpacity=0.2,
            weight=2
        ).add_to(map_object)

        print(f"✓ Circle added: {site_name}")
        print(f"    Radius: {radius/1000:.0f} km")
        print(f"    Success Rate: {stats['success_rate']:.1f}%")


def add_legend(map_object: folium.Map) -> None:
    """
    Add legend to map.

    Args:
        map_object: Folium Map object
    """
    print("\nAdding legend to map...")

    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 250px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px; border-radius: 5px;">

    <b>SpaceX Launch Sites - Landing Success Rates</b>
    <hr style="margin: 10px 0;">

    <p><i class="fa fa-circle" style="color:#2ca02c"></i> High Success (>66%)</p>
    <p><i class="fa fa-circle" style="color:#1f77b4"></i> Medium Success (33-66%)</p>
    <p><i class="fa fa-circle" style="color:#d62728"></i> Low Success (<33%)</p>

    <hr style="margin: 10px 0;">
    <p><small><b>Circle Size:</b> Proportional to number of launches</small></p>
    <p><small><b>Colors:</b> Based on landing success rate</small></p>

    <p style="margin-top: 10px; font-size: 12px; color: #666;">
    <i>IBM Data Science Capstone Project</i>
    </p>
    </div>
    '''

    map_object.get_root().html.add_child(folium.Element(legend_html))
    print("✓ Legend added to map")


def add_map_title(map_object: folium.Map) -> None:
    """
    Add title to map.

    Args:
        map_object: Folium Map object
    """
    title_html = '''
    <div style="position: fixed;
                top: 10px; left: 50px; width: auto; height: auto;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:16px; padding: 10px; border-radius: 5px;
                font-weight: bold; text-align: center;">
    SpaceX Falcon 9 Launch Sites Map
    </div>
    '''

    map_object.get_root().html.add_child(folium.Element(title_html))
    print("✓ Title added to map")


# ══════════════════════════════════════════════════════════════════════════════
# File Export
# ══════════════════════════════════════════════════════════════════════════════

def save_map(map_object: folium.Map, output_path: str) -> None:
    """
    Save map as HTML file.

    Args:
        map_object: Folium Map object
        output_path: Path to output HTML file
    """
    print("\n" + "="*80)
    print("SAVING MAP TO HTML")
    print("="*80)

    try:
        map_object.save(output_path)
        print(f"✓ Map saved successfully")
        print(f"  Output: {output_path}")
        print(f"  Size: {len(map_object._repr_html_()) / 1024:.1f} KB")

    except Exception as e:
        print(f"✗ Error saving map: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main execution function for interactive map creation.
    """
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "SpaceX Falcon 9 - Interactive Launch Sites Map".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    start_time = datetime.now()
    print(f"\nExecution started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data(DATA_PATH)
    if df is None:
        print("✗ Failed to load data. Exiting.")
        return

    # Prepare site statistics
    site_stats = prepare_site_statistics(df)
    if len(site_stats) == 0:
        print("✗ No launch site data found. Exiting.")
        return

    # Create base map
    map_object = create_base_map()

    # Add visual elements
    add_site_circles(map_object, site_stats)
    add_site_markers(map_object, site_stats)
    add_map_title(map_object)
    add_legend(map_object)

    # Save map
    save_map(map_object, OUTPUT_PATH)

    # Summary
    print("\n" + "="*80)
    print("MAP CREATION SUMMARY")
    print("="*80)
    print(f"✓ Interactive map created successfully")
    print(f"  Launch sites mapped: {len(site_stats)}")
    print(f"  Total launches: {sum(s['total_launches'] for s in site_stats.values())}")
    print(f"  Output file: {OUTPUT_PATH}")
    print(f"\nTo view the map:")
    print(f"  1. Open: {OUTPUT_PATH}")
    print(f"  2. View in web browser")
    print(f"  3. Interact with markers and circles")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n✓ Execution completed in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
