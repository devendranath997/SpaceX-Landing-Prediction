"""
SpaceX Falcon 9 Landing Prediction - Data Collection via Web Scraping
IBM Data Science Professional Certificate Capstone Project

This module scrapes SpaceX Falcon 9 launch data from Wikipedia's official launch list.
It extracts tabular data from the "List of Falcon 9 and Falcon Heavy launches" article,
cleans the data (removing footnotes, handling missing values), and exports to CSV.

Key Features:
- HTTP GET requests to Wikipedia
- HTML parsing with BeautifulSoup
- Extraction of wikitable data (rows and columns)
- Data cleaning (footnote removal, date formatting, missing value handling)
- Validation and export to structured CSV format

Author: Devendra Nath (devendranath997)
Date: 2024
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
from typing import List, Tuple, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches"
OUTPUT_DIR = "../data/raw"
REQUEST_TIMEOUT = 10  # seconds
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


# ══════════════════════════════════════════════════════════════════════════════
# Web Scraping Functions
# ══════════════════════════════════════════════════════════════════════════════

def fetch_wikipedia_page(url: str) -> Optional[BeautifulSoup]:
    """
    Fetch and parse Wikipedia page containing Falcon 9 launch data.

    Args:
        url: Wikipedia URL to scrape

    Returns:
        BeautifulSoup object if successful, None otherwise
    """
    print("\n" + "="*80)
    print("FETCHING WIKIPEDIA PAGE")
    print("="*80)
    print(f"URL: {url}")

    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        print("✓ Successfully fetched and parsed Wikipedia page")
        return soup

    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching page: {e}")
        return None


def extract_wikitable_data(soup: BeautifulSoup) -> Optional[pd.DataFrame]:
    """
    Extract data from wikitable in the parsed HTML.

    Args:
        soup: BeautifulSoup parsed HTML object

    Returns:
        DataFrame with raw extracted data
    """
    print("\n" + "="*80)
    print("EXTRACTING WIKITABLE DATA")
    print("="*80)

    try:
        # Find all wikitables
        tables = soup.find_all('table', {'class': 'wikitable'})
        print(f"Found {len(tables)} wikitable(s) on the page")

        if len(tables) == 0:
            print("✗ No wikitables found")
            return None

        # Use the first comprehensive table (usually contains launch data)
        table = tables[0]
        print(f"✓ Parsing primary wikitable")

        # Extract headers
        headers = []
        header_row = table.find('tr')
        if header_row:
            for th in header_row.find_all(['th', 'td']):
                headers.append(th.get_text(strip=True))

        print(f"  Found {len(headers)} columns: {headers[:5]}...")

        # Extract rows
        rows = []
        for tr in table.find_all('tr')[1:]:  # Skip header row
            cells = [td.get_text(strip=True) for td in tr.find_all(['td'])]
            if len(cells) > 0:
                rows.append(cells)

        print(f"✓ Extracted {len(rows)} data rows")

        # Create DataFrame
        if len(headers) > 0 and len(rows) > 0:
            df = pd.DataFrame(rows, columns=headers[:len(rows[0])])
            return df
        else:
            print("✗ Unable to create DataFrame from extracted data")
            return None

    except Exception as e:
        print(f"✗ Error extracting wikitable data: {e}")
        return None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean extracted data: remove footnotes, handle missing values, format dates.

    Args:
        df: Raw extracted DataFrame

    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*80)
    print("CLEANING DATA")
    print("="*80)

    # Create a copy to avoid modifying original
    df_clean = df.copy()

    print(f"Initial shape: {df_clean.shape}")

    # Remove footnote citations (e.g., [1], [2], etc.)
    print("  Removing footnote citations...")
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(
                lambda x: re.sub(r'\[\d+\]', '', x) if isinstance(x, str) else x
            )

    # Handle empty strings and replace with NaN
    print("  Handling empty strings...")
    df_clean = df_clean.replace('', pd.NA)

    # Remove rows with all NaN values
    df_clean = df_clean.dropna(how='all')
    print(f"  Removed empty rows. Shape: {df_clean.shape}")

    # Strip whitespace from all string columns
    print("  Stripping whitespace...")
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )

    # Standardize column names
    print("  Standardizing column names...")
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace('[', '').str.replace(']', '')

    print(f"✓ Data cleaning completed. Final shape: {df_clean.shape}")
    print(f"  Missing values per column:\n{df_clean.isnull().sum()}")

    return df_clean


def validate_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate scraped data for quality and completeness.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    print("\n" + "="*80)
    print("VALIDATING DATA")
    print("="*80)

    validations = []

    # Check if DataFrame is empty
    if df.shape[0] == 0:
        return False, "DataFrame is empty"
    validations.append(f"✓ Non-empty DataFrame: {df.shape[0]} rows")

    # Check if DataFrame has columns
    if df.shape[1] == 0:
        return False, "DataFrame has no columns"
    validations.append(f"✓ Column count: {df.shape[1]}")

    # Check for minimum rows (should have at least 50 Falcon 9 launches)
    if df.shape[0] < 50:
        validations.append(f"⚠ Warning: Only {df.shape[0]} rows (expected >50)")
    else:
        validations.append(f"✓ Sufficient row count: {df.shape[0]}")

    # Print validation results
    for validation in validations:
        print(f"  {validation}")

    return True, "Data validation passed"


def export_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Export cleaned data to CSV file.

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
    Main execution function for web scraping data collection.
    Orchestrates the entire scraping workflow.
    """
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "SpaceX Falcon 9 Landing Prediction - Web Scraping Collection".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    start_time = datetime.now()
    print(f"\nExecution started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Fetch Wikipedia page
    soup = fetch_wikipedia_page(WIKIPEDIA_URL)
    if soup is None:
        print("✗ Failed to fetch Wikipedia page. Exiting.")
        return

    # Extract wikitable data
    raw_df = extract_wikitable_data(soup)
    if raw_df is None:
        print("✗ Failed to extract wikitable data. Exiting.")
        return

    # Clean data
    clean_df = clean_data(raw_df)

    # Validate data
    is_valid, message = validate_data(clean_df)
    print(f"  {message}")

    if not is_valid:
        print("✗ Data validation failed. Exiting.")
        return

    # Export to CSV
    output_path = f"{OUTPUT_DIR}/spacex_launches_wikipedia.csv"
    export_data(clean_df, output_path)

    # Summary
    print("\n" + "="*80)
    print("DATA COLLECTION SUMMARY")
    print("="*80)
    print(f"Total launches scraped: {clean_df.shape[0]}")
    print(f"Total features: {clean_df.shape[1]}")
    print(f"\nColumn names:")
    for i, col in enumerate(clean_df.columns, 1):
        print(f"  {i}. {col}")

    print(f"\nFirst 3 rows:")
    print(clean_df.head(3).to_string())

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n✓ Execution completed in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
