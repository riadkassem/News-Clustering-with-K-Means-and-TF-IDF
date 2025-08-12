# News Clustering with NLP and Machine Learning

## Project Overview
This project performs unsupervised clustering on news articles using natural language processing (NLP) and machine learning techniques to automatically group similar articles based on their content.

## Features

### Core Functionality
- Text preprocessing (tokenization, stemming, stopword removal)
- TF-IDF vectorization and PCA dimensionality reduction
- K-Means and Agglomerative clustering implementations
- Elbow method for optimal cluster determination

### Visualization
- 2D scatter plots of article clusters
- Interactive cluster exploration
- Silhouette analysis for cluster quality

### Utilities
- Similar article recommendation system
- Cluster analysis and interpretation

## Installation

```bash
git clone https://github.com/yourusername/news-clustering.git
cd news-clustering
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
```
### Implementation

The notebook (news_clustering.ipynb) contains:

    1. Data Preparation

        - Loading and balancing the dataset

        - Text cleaning and normalization

    2. Feature Engineering

        - TF-IDF vectorization

        - PCA for dimensionality reduction

    3. Clustering

        - K-Means with elbow method optimization

        - Cluster visualization and analysis

    4. Utilities

        - Finding similar articles

        - Cluster interpretation

## Requirements

    - Python 3.7+

    - pandas, numpy, scikit-learn

    - matplotlib, seaborn, scikit-learn

    - nltk

## Future Work

    - Implement DBSCAN/HDBSCAN clustering

    - Detect Anomalies in Credit Card Transactions