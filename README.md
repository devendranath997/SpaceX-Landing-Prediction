# SpaceX Falcon 9 First-Stage Landing Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly Dash](https://img.shields.io/badge/Plotly_Dash-2.14+-3F4F75?style=flat&logo=plotly&logoColor=white)](https://dash.plotly.com)
[![Folium](https://img.shields.io/badge/Folium-0.14+-77B829?style=flat&logo=leaflet&logoColor=white)](https://python-visualization.github.io/folium/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Can we predict whether a SpaceX Falcon 9 first stage will land successfully?**
>
> This end-to-end data science project collects, cleans, explores, visualizes, and models SpaceX launch data to predict booster landing outcomes — helping estimate launch costs and supporting mission planning decisions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Methodology](#methodology)
- [Results](#results)
- [Interactive Dashboard](#interactive-dashboard)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

SpaceX has revolutionized the space industry by developing reusable Falcon 9 first-stage boosters, reducing launch costs from ~$165M (competitors) to ~$62M. Predicting whether a booster will successfully land is critical for:

- **Cost estimation** for competitive bidding against SpaceX
- **Mission planning** and risk assessment
- **Understanding factors** that influence landing success

This project applies the full data science methodology — from data collection through predictive modeling — to answer these questions using real SpaceX launch data.

### Research Questions

1. How do payload mass, launch site, and orbit type affect first-stage landing success?
2. Does launch success improve over time as SpaceX gains operational experience?
3. Which launch sites and mission profiles have the highest success rates?
4. Can historical data be used to predict the probability of a successful landing?

---

## Key Findings

- **Landing success has improved dramatically over time**, from ~33% in early missions to near 100% in recent launches, demonstrating SpaceX's learning curve
- **Launch site matters**: KSC LC-39A handles heavier payloads and achieves high success rates
- **Orbit type influences success**: GTO orbits show lower success rates compared to LEO and SSO
- **Best predictive model**: Support Vector Machine (SVM) and Decision Tree classifiers achieved the highest accuracy (~83%) in predicting landing outcomes
- **Payload mass correlation**: Lighter payloads generally have higher success rates, though recent missions show improvement across all payload ranges

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.9+ |
| **Data Collection** | Requests, BeautifulSoup4, SpaceX REST API |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Geospatial** | Folium |
| **Database** | SQLite3 |
| **Machine Learning** | scikit-learn (Logistic Regression, SVM, Decision Tree, KNN) |
| **Dashboard** | Plotly Dash |

---

## Project Structure

```
SpaceX-Landing-Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   ├── 01_data_collection_api.py          # SpaceX API data collection
│   ├── 02_data_collection_webscraping.py  # Wikipedia web scraping
│   ├── 03_data_wrangling.py               # Data cleaning & feature engineering
│   ├── 04_eda_visualization.py            # Exploratory analysis & plots
│   ├── 05_eda_sql.py                      # SQL-based analysis
│   ├── 06_interactive_map_folium.py       # Interactive launch site map
│   └── 07_predictive_analysis.py          # ML classification models
├── src/
│   └── spacex_dash_app.py                 # Plotly Dash interactive dashboard
├── data/                                  # Generated datasets (CSV, DB)
├── docs/
│   └── SpaceX_Capstone_Presentation.pdf   # Final presentation
└── images/                                # Generated visualizations
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/devendranath997/SpaceX-Landing-Prediction.git
cd SpaceX-Landing-Prediction

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

Execute the scripts in order to reproduce the full analysis:

```bash
# Step 1: Collect data
python notebooks/01_data_collection_api.py
python notebooks/02_data_collection_webscraping.py

# Step 2: Clean and prepare data
python notebooks/03_data_wrangling.py

# Step 3: Exploratory analysis
python notebooks/04_eda_visualization.py
python notebooks/05_eda_sql.py
python notebooks/06_interactive_map_folium.py

# Step 4: Train and evaluate ML models
python notebooks/07_predictive_analysis.py

# Step 5: Launch interactive dashboard
python src/spacex_dash_app.py
# Open http://127.0.0.1:8050 in your browser
```

---

## Methodology

### 1. Data Collection
- **SpaceX REST API** (`api.spacexdata.com/v4`): Retrieved 90+ launch records with rocket specs, launchpad details, and landing outcomes
- **Web Scraping**: Parsed Wikipedia's Falcon 9 launch tables using BeautifulSoup for supplementary data

### 2. Data Wrangling
- Handled 28.9% missing values in `LandingPad` field
- Encoded categorical variables (Orbit, LaunchSite, Outcome)
- Created binary target variable: `Class` (1 = Successful landing, 0 = Failed landing)
- Engineered time-based and payload-category features

### 3. Exploratory Data Analysis
- **Visualization (Matplotlib/Seaborn)**: Scatter plots, bar charts, and line charts revealing trends in payload mass, orbit type, and temporal success patterns
- **SQL Analysis (SQLite)**: Aggregation queries for success rates by site, orbit, and year
- **Geospatial (Folium)**: Interactive maps showing launch site locations, proximity to infrastructure, and color-coded outcomes

### 4. Interactive Dashboard
- Built with **Plotly Dash** featuring dropdown filters for launch sites, payload range sliders, and real-time pie/scatter charts

### 5. Predictive Modeling
- Trained and tuned 4 classification models using **GridSearchCV**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Decision Tree Classifier
  - K-Nearest Neighbors (KNN)
- Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrices

---

## Results

### Model Performance Comparison

| Model | Test Accuracy | Best Hyperparameters |
|-------|:------------:|----------------------|
| **Decision Tree** | **~83%** | max_depth=5, min_samples_split=5 |
| **SVM** | **~83%** | C=1.0, kernel=rbf, gamma=scale |
| Logistic Regression | ~78% | C=0.1, solver=lbfgs |
| KNN | ~78% | n_neighbors=7, weights=distance |

> **Best Models**: Decision Tree and SVM achieved the highest accuracy, with SVM showing more robust generalization on unseen data.

---

## Interactive Dashboard

The Plotly Dash application provides real-time exploration of launch data:

- **Site Selector**: Filter by individual launch site or view all sites
- **Payload Slider**: Focus on specific payload mass ranges
- **Pie Chart**: Success/failure distribution by site
- **Scatter Plot**: Payload mass vs. landing outcome, colored by booster version

Launch the dashboard:
```bash
python src/spacex_dash_app.py
```

---

## Author

**Devendranath Maganti**
- M.S. Applied Statistics (Data Science) — Colorado State University
- [LinkedIn](https://www.linkedin.com/in/devendranath-maganti/)
- [GitHub](https://github.com/devendranath997)

---

## Acknowledgments

- **IBM Data Science Professional Certificate** — Coursera
- **SpaceX** — Public API and open launch data
- **IBM Developer Skills Network** — Course labs and guidance

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
