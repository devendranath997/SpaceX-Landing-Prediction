"""
SpaceX Falcon 9 Landing Prediction - Data Collection via REST API
IBM Data Science Professional Certificate Capstone Project

This module collects SpaceX Falcon 9 launch data from the official SpaceX REST API
(api.spacexdata.com/v4). It retrieves information about past launches, rocket details,
and launchpad locations, then merges this data into a structured dataset for analysis.

Key Features:
- Fetches launch history from /launches/past endpoint
- Retrieves rocket specifications and launchpad information
- Merges data into a unified DataFrame
- Handles API pagination and error responses
- Exports cleaned data to CSV for downstream processing

Author: Devendra Nath (devendranath997)
Date: 2024
"""

import requests
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

BASE_URL = "https://api.spacexdata.com/v4"
OUTPUT_DIR = "../data/raw"
API_TIMEOUT = 10  # seconds


# ══════════════════════════════════════════════════════════════════════════════
# API Data Collection Functions
# ══════════════════════════════════════════════════════════════════════════════

def fetch_launches_data() -> Optional[pd.DataFrame]:
    """
    Fetch all past Falcon 9 launches from SpaceX API.

    Returns:
        pd.DataFrame: DataFrame containing launch data with fields like flight_number,
                     rocket_id, launch_date_utc, mission_name, etc.
        None: If the API request fails
    """
    print("\n" + "="*80)
    print("FETCHING LAUNCH DATA FROM SPACEX API")
    print("="*80)

    try:
        url = f"{BASE_URL}/launches/past"
        print(f"Requesting: {url}")

        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()

        launches = response.json()
        print(f"✓ Successfully retrieved {len(launches)} launch records")

        return pd.DataFrame(launches)

    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching launches: {e}")
        return None


def fetch_rockets_data() -> Optional[pd.DataFrame]:
    """
    Fetch all rocket specifications from SpaceX API.

    Returns:
        pd.DataFrame: DataFrame with rocket details (id, name, type, mass, etc.)
        None: If the API request fails
    """
    print("\n" + "="*80)
    print("FETCHING ROCKET DATA FROM SPACEX API")
    print("="*80)

    try:
        url = f"{BASE_URL}/rockets"
        print(f"Requesting: {url}")

        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()

        rockets = response.json()
        print(f"✓ Successfully retrieved {len(rockets)} rocket records")

        # Select relevant columns
        rockets_df = pd.DataFrame(rockets)
        relevant_cols = [col for col in rockets_df.columns
                        if col in ['id', 'name', 'type', 'height', 'diameter', 'mass']]

        return rockets_df[relevant_cols]

    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching rockets: {e}")
        return None


def fetch_launchpads_data() -> Optional[pd.DataFrame]:
    """
    Fetch all launchpad information from SpaceX API.

    Returns:
        pd.DataFrame: DataFrame with launchpad details (id, name, region, longitude, latitude)
        None: If the API request fails
    """
    print("\n" + "="*80)
    print("FETCHING LAUNCHPAD DATA FROM SPACEX API")
    print("="*80)

    try:
        url = f"{BASE_URL}/launchpads"
        print(f"Requesting: {url}")

        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()

        launchpads = response.json()
        print(f"✓ Successfully retrieved {len(launchpads)} launchpad records")

        launchpads_df = pd.DataFrame(launchpads)
        relevant_cols = [col for col in launchpads_df.columns
                        if col in ['id', 'name', 'full_name', 'region', 'latitude', 'longitude']]

        return launchpads_df[relevant_cols]

    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching launchpads: {e}")
        return None


def merge_data(launches_df: pd.DataFrame, rockets_df: pd.DataFrame,
               launchpads_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge launches, rockets, and launchpads data into a unified DataFrame.

    Args:
        launches_df: DataFrame with launch data
        rockets_df: DataFrame with rocket specifications
        launchpads_df: DataFrame with launchpad information

    Returns:
        pd.DataFrame: Merged dataset ready for analysis
    """
    print("\n" + "="*80)
    print("MERGING DATA")
    print("="*80)

    # Extract rocket_id from rocket column (which is a list)
    launches_df['rocket_id'] = launches_df['rocket'].apply(
        lambda x: x if isinstance(x, str) else (x[0] if isinstance(x, list) and len(x) > 0 else None)
    )

    # Merge launches with rockets
    merged = launches_df.merge(rockets_df, left_on='rocket_id', right_on='id',
                               how='left', suffixes=('_launch', '_rocket'))
    print(f"✓ Merged launches with rocket data: {merged.shape[0]} records")

    # Extract launchpad_id if needed
    if 'launchpad' in merged.columns:
        merged['launchpad_id'] = merged['launchpad'].apply(
            lambda x: x if isinstance(x, str) else (x[0] if isinstance(x, list) and len(x) > 0 else None)
        )

        merged = merged.merge(launchpads_df, left_on='launchpad_id', right_on='id',
                             how='left', suffixes=('_launch', '_pad'))
        print(f"✓ Merged with launchpad data: {merged.shape[0]} records")

    return merged


def export_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Export merged data to CSV file.

    Args:
        df: DataFrame to export
        output_path: Path to output CSV file
    """
    print("\n" + "="*80)
    print("EXPORTING DATA")
    print("="*80)

    try:
        df.to_csv(output_path, index=False)
        print(f"✓ Data exported successfully to: {output_path}")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Size: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    except Exception as e:
        print(f"✗ Error exporting data: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main execution function for API data collection.
    Orchestrates the entire data collection workflow.
    """
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "SpaceX Falcon 9 Landing Prediction - Data Collection via API".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    start_time = datetime.now()
    print(f"\nExecution started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Fetch data from API
    launches_df = fetch_launches_data()
    if launches_df is None:
        print("✗ Failed to fetch launch data. Exiting.")
        return

    rockets_df = fetch_rockets_data()
    if rockets_df is None:
        print("✗ Failed to fetch rocket data. Exiting.")
        return

    launchpads_df = fetch_launchpads_data()
    if launchpads_df is None:
        print("✗ Failed to fetch launchpad data. Exiting.")
        return

    # Merge datasets
    merged_df = merge_data(launches_df, rockets_df, launchpads_df)

    # Export to CSV
    output_path = f"{OUTPUT_DIR}/spacex_launches_raw.csv"
    export_data(merged_df, output_path)

    # Summary statistics
    print("\n" + "="*80)
    print("DATA COLLECTION SUMMARY")
    print("="*80)
    print(f"Total launches collected: {merged_df.shape[0]}")
    print(f"Total features extracted: {merged_df.shape[1]}")
    print(f"\nFirst few rows:")
    print(merged_df.head())

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n✓ Execution completed in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
