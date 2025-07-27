import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

INPUT_DATA_FILE = "mobility_survey_data.csv"
TRANSITION_MATRIX_FILE = "transition_matrix_results.csv"
CLUSTER_RESULTS_FILE = "vulnerability_clusters.csv"
VISUALIZATION_FILE = "mobility_analysis_plots.png"

def simulate_dataset():
    np.random.seed(42)
    occupations = ['Agriculture', 'Manufacturing', 'Services', 'Professional']
    
    data = pd.DataFrame({
        'respondent_id': [f"ID_{x:03d}" for x in range(1, 81)],
        'mother_occupation': np.random.choice(occupations, 80, p=[0.3, 0.25, 0.35, 0.1]),
        'current_occupation': np.random.choice(occupations, 80, p=[0.2, 0.2, 0.45, 0.15]),
        'income_cohort': np.random.choice(['Low', 'Middle', 'High'], 80, p=[0.4, 0.35, 0.25]),
        'savings_volatility': np.abs(np.random.normal(0.5, 0.15, 80)),
        'financial_access': np.random.beta(2, 5, 80),
        'education_years': np.random.randint(5, 18, 80)
    })
    
    data.to_csv(INPUT_DATA_FILE, index=False)
    return data

def analyze_transitions(df):
    transition_matrix = pd.crosstab(df['mother_occupation'], df['current_occupation'])
    transition_matrix.to_csv(TRANSITION_MATRIX_FILE)
    
    chi2, pval = stats.chi2_contingency(transition_matrix)[:2]
    
    print(f"Transition matrix saved to {TRANSITION_MATRIX_FILE}")
    print(f"Chi-squared p-value: {pval:.4f}")
    
    return transition_matrix

def perform_clustering(df):
    features = ['savings_volatility', 'financial_access', 'education_years']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['vulnerability_cluster'] = kmeans.fit_predict(scaled_data)
    
    df.to_csv(CLUSTER_RESULTS_FILE, index=False)
    print(f"Cluster results saved to {CLUSTER_RESULTS_FILE}")
    
    return df

def create_visualizations(df):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(pd.crosstab(df['mother_occupation'], df['current_occupation']), 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Occupational Transitions')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(x='vulnerability_cluster', y='savings_volatility', data=df)
    plt.title('Savings Volatility by Cluster')
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(x='financial_access', y='education_years', 
                   hue='vulnerability_cluster', data=df)
    plt.title('Financial Access vs Education')
    
    plt.tight_layout()
    plt.savefig(VISUALIZATION_FILE)
    print(f"Visualizations saved to {VISUALIZATION_FILE}")

def main():
    print(f"Starting analysis with input from {INPUT_DATA_FILE}")
    
    data = simulate_dataset()
    transition_results = analyze_transitions(data)
    clustered_data = perform_clustering(data)
    create_visualizations(clustered_data)
    
    print(f"Analysis completed. Output files:")
    print(f"- {TRANSITION_MATRIX_FILE}")
    print(f"- {CLUSTER_RESULTS_FILE}")
    print(f"- {VISUALIZATION_FILE}")

if __name__ == "__main__":
    main()
