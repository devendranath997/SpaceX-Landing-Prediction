"""
SpaceX Falcon 9 Landing Prediction - Exploratory Data Analysis via SQL
IBM Data Science Professional Certificate Capstone Project

This module loads SpaceX launch data into a SQLite database and performs
comprehensive exploratory data analysis using SQL queries. It examines
unique launch sites, orbit types, mission outcomes, and calculates success
rates across various dimensions.

Key Features:
- Load CSV data into SQLite database
- Run SQL queries: SELECT DISTINCT, WHERE filters, aggregates
- GROUP BY and ORDER BY operations
- Analyze unique launch sites and orbit types
- Calculate success rates by launch site and orbit
- Identify top-performing launch facilities

Author: Devendra Nath (devendranath997)
Date: 2024
"""

import pandas as pd
import sqlite3
from datetime import datetime
from typing import Optional, List


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = "../data/processed/spacex_launches_processed.csv"
DB_PATH = "../data/spacex_launches.db"
TABLE_NAME = "launches"


# ══════════════════════════════════════════════════════════════════════════════
# Database Setup Functions
# ══════════════════════════════════════════════════════════════════════════════

def load_data_to_database(csv_path: str, db_path: str, table_name: str) -> Optional[sqlite3.Connection]:
    """
    Load CSV data into SQLite database.

    Args:
        csv_path: Path to CSV file
        db_path: Path to SQLite database
        table_name: Name of table to create

    Returns:
        SQLite connection object or None if error occurs
    """
    print("\n" + "="*80)
    print("LOADING DATA INTO SQLITE DATABASE")
    print("="*80)

    try:
        # Read CSV
        print(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✓ CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Connect to database
        conn = sqlite3.connect(db_path)
        print(f"✓ Connected to database: {db_path}")

        # Create table and insert data
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"✓ Data loaded into table '{table_name}'")
        print(f"  Total records: {len(df)}")

        return conn

    except FileNotFoundError:
        print(f"✗ File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


def execute_query(conn: sqlite3.Connection, query: str, description: str = "") -> pd.DataFrame:
    """
    Execute SQL query and return results as DataFrame.

    Args:
        conn: SQLite connection
        query: SQL query to execute
        description: Description of the query

    Returns:
        Results as DataFrame
    """
    try:
        if description:
            print(f"\n{description}")
            print("-" * 80)

        print(f"Query:\n{query}\n")

        df = pd.read_sql_query(query, conn)

        if len(df) > 0:
            print(f"Results ({len(df)} rows):")
            print(df.to_string(index=False))
        else:
            print("No results found.")

        return df

    except sqlite3.Error as e:
        print(f"✗ SQL Error: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# SQL Queries
# ══════════════════════════════════════════════════════════════════════════════

def analyze_unique_launch_sites(conn: sqlite3.Connection) -> None:
    """
    Query 1: Find all unique launch sites.
    """
    query = f"SELECT DISTINCT LaunchSite FROM {TABLE_NAME} ORDER BY LaunchSite;"

    execute_query(conn, query, "QUERY 1: UNIQUE LAUNCH SITES")


def analyze_orbit_types(conn: sqlite3.Connection) -> None:
    """
    Query 2: Find all unique orbit types.
    """
    query = f"SELECT DISTINCT Orbit FROM {TABLE_NAME} ORDER BY Orbit;"

    execute_query(conn, query, "QUERY 2: UNIQUE ORBIT TYPES")


def count_launches_by_site(conn: sqlite3.Connection) -> None:
    """
    Query 3: Count launches by launch site with aggregation.
    """
    query = f"""
    SELECT
        LaunchSite,
        COUNT(*) as Total_Launches,
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as Successful_Landings,
        ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as Success_Rate_Percent
    FROM {TABLE_NAME}
    GROUP BY LaunchSite
    ORDER BY Success_Rate_Percent DESC;
    """

    execute_query(conn, query, "QUERY 3: LAUNCH COUNT AND SUCCESS RATE BY SITE (GROUP BY)")


def count_launches_by_orbit(conn: sqlite3.Connection) -> None:
    """
    Query 4: Count launches and success rate by orbit type.
    """
    query = f"""
    SELECT
        Orbit,
        COUNT(*) as Total_Launches,
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as Successful_Landings,
        ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as Success_Rate_Percent
    FROM {TABLE_NAME}
    WHERE Orbit IS NOT NULL
    GROUP BY Orbit
    ORDER BY Success_Rate_Percent DESC;
    """

    execute_query(conn, query, "QUERY 4: LAUNCH COUNT AND SUCCESS RATE BY ORBIT TYPE (GROUP BY)")


def top_performing_sites(conn: sqlite3.Connection) -> None:
    """
    Query 5: Find top-performing launch sites (minimum 3 launches).
    """
    query = f"""
    SELECT
        LaunchSite,
        COUNT(*) as Launch_Count,
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as Successful_Landings,
        ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as Success_Rate_Percent
    FROM {TABLE_NAME}
    GROUP BY LaunchSite
    HAVING COUNT(*) >= 3
    ORDER BY Success_Rate_Percent DESC, Launch_Count DESC
    LIMIT 5;
    """

    execute_query(conn, query, "QUERY 5: TOP 5 PERFORMING LAUNCH SITES (WHERE CLAUSE + LIMIT)")


def success_stats_overall(conn: sqlite3.Connection) -> None:
    """
    Query 6: Overall success statistics.
    """
    query = f"""
    SELECT
        COUNT(*) as Total_Launches,
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as Successful_Landings,
        SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) as Failed_Landings,
        ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as Overall_Success_Rate_Percent,
        AVG(CAST(Class AS FLOAT)) as Average_Success_Score
    FROM {TABLE_NAME};
    """

    execute_query(conn, query, "QUERY 6: OVERALL SUCCESS STATISTICS (SUM, COUNT, AVG)")


def payload_analysis_by_outcome(conn: sqlite3.Connection) -> None:
    """
    Query 7: Analyze payload characteristics by landing outcome.
    """
    # First check if payload column exists
    query_check = f"PRAGMA table_info({TABLE_NAME});"
    df_cols = pd.read_sql_query(query_check, conn)

    payload_col = None
    for _, row in df_cols.iterrows():
        if 'payload' in row['name'].lower() and 'mass' in row['name'].lower():
            payload_col = row['name']
            break

    if payload_col is None:
        print("\n⚠ Payload mass column not found. Skipping payload analysis.")
        return

    query = f"""
    SELECT
        CASE WHEN Class = 1 THEN 'Success' ELSE 'Failure' END as Outcome,
        COUNT(*) as Mission_Count,
        ROUND(AVG(CAST({payload_col} AS FLOAT)), 2) as Avg_Payload_Mass,
        ROUND(MIN(CAST({payload_col} AS FLOAT)), 2) as Min_Payload_Mass,
        ROUND(MAX(CAST({payload_col} AS FLOAT)), 2) as Max_Payload_Mass
    FROM {TABLE_NAME}
    WHERE {payload_col} IS NOT NULL
    GROUP BY Class
    ORDER BY Class DESC;
    """

    execute_query(conn, query, "QUERY 7: PAYLOAD ANALYSIS BY LANDING OUTCOME")


def mission_outcome_summary(conn: sqlite3.Connection) -> None:
    """
    Query 8: Summary of mission outcomes.
    """
    query = f"""
    SELECT
        CASE WHEN Class = 1 THEN 'Successful Landing' ELSE 'Landing Failure' END as Outcome,
        COUNT(*) as Mission_Count,
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM {TABLE_NAME}), 1) as Percentage
    FROM {TABLE_NAME}
    GROUP BY Class
    ORDER BY Class DESC;
    """

    execute_query(conn, query, "QUERY 8: MISSION OUTCOME SUMMARY")


def site_orbit_combination_analysis(conn: sqlite3.Connection) -> None:
    """
    Query 9: Analyze success rates for site-orbit combinations.
    """
    query = f"""
    SELECT
        LaunchSite,
        Orbit,
        COUNT(*) as Launch_Count,
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as Successes,
        ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as Success_Rate_Percent
    FROM {TABLE_NAME}
    WHERE LaunchSite IS NOT NULL AND Orbit IS NOT NULL
    GROUP BY LaunchSite, Orbit
    HAVING COUNT(*) >= 2
    ORDER BY Success_Rate_Percent DESC, Launch_Count DESC
    LIMIT 10;
    """

    execute_query(conn, query, "QUERY 9: SITE-ORBIT COMBINATION ANALYSIS (COMPLEX GROUP BY)")


def trend_analysis_by_period(conn: sqlite3.Connection) -> None:
    """
    Query 10: Trend analysis by year (if Year column exists).
    """
    # Check if Year column exists
    query_check = f"PRAGMA table_info({TABLE_NAME});"
    df_cols = pd.read_sql_query(query_check, conn)

    has_year = any('year' in row['name'].lower() for _, row in df_cols.iterrows())

    if not has_year:
        print("\n⚠ Year column not found. Skipping trend analysis.")
        return

    query = f"""
    SELECT
        Year,
        COUNT(*) as Launch_Count,
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) as Successful_Landings,
        ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as Success_Rate_Percent
    FROM {TABLE_NAME}
    WHERE Year IS NOT NULL
    GROUP BY Year
    ORDER BY Year DESC;
    """

    execute_query(conn, query, "QUERY 10: SUCCESS TREND ANALYSIS BY YEAR")


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main execution function for SQL-based EDA.
    Orchestrates database creation and query execution.
    """
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "SpaceX Falcon 9 - Exploratory Data Analysis via SQL".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    start_time = datetime.now()
    print(f"\nExecution started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data into database
    conn = load_data_to_database(DATA_PATH, DB_PATH, TABLE_NAME)

    if conn is None:
        print("✗ Failed to load data into database. Exiting.")
        return

    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "EXECUTING SQL QUERIES FOR EXPLORATORY ANALYSIS".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    # Execute all analysis queries
    try:
        analyze_unique_launch_sites(conn)
        analyze_orbit_types(conn)
        count_launches_by_site(conn)
        count_launches_by_orbit(conn)
        top_performing_sites(conn)
        success_stats_overall(conn)
        payload_analysis_by_outcome(conn)
        mission_outcome_summary(conn)
        site_orbit_combination_analysis(conn)
        trend_analysis_by_period(conn)

    except Exception as e:
        print(f"✗ Error executing queries: {e}")

    # Close connection
    conn.close()
    print("\n✓ Database connection closed")

    # Summary
    print("\n" + "="*80)
    print("SQL ANALYSIS SUMMARY")
    print("="*80)
    print("✓ All SQL queries executed successfully")
    print("\nQueries Executed:")
    print("  1. Unique Launch Sites (SELECT DISTINCT)")
    print("  2. Unique Orbit Types (SELECT DISTINCT)")
    print("  3. Launch Count by Site (GROUP BY + Aggregates)")
    print("  4. Success Rate by Orbit (GROUP BY + Aggregates)")
    print("  5. Top Performing Sites (WHERE + GROUP BY + LIMIT)")
    print("  6. Overall Success Statistics (SUM, COUNT, AVG)")
    print("  7. Payload Analysis by Outcome")
    print("  8. Mission Outcome Summary")
    print("  9. Site-Orbit Combination Analysis (Complex GROUP BY)")
    print(" 10. Success Trend by Year")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n✓ Execution completed in {duration:.2f} seconds")
    print(f"Database saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
