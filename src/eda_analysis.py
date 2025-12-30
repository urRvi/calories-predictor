import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    print("Loading dataset...")
    df = pd.read_csv('data/calories.csv')
    
    print("\nDataset Info:")
    df.info()
    
    print("\nDataset Description:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Correlation
    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    print("\nCorrelation Matrix:")
    print(corr)
    
    # Save plots
    print("\nSaving plots...")
    os.makedirs('notebooks/plots', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('notebooks/plots/correlation_heatmap.png')
    print("Saved correlation_heatmap.png")
    
    # Pairplot (might be slow for large datasets, so maybe sample it)
    if len(df) > 1000:
        sample_df = df.sample(1000, random_state=42)
    else:
        sample_df = df
        
    sns.pairplot(sample_df)
    plt.savefig('notebooks/plots/pairplot.png')
    print("Saved pairplot.png")

if __name__ == "__main__":
    run_eda()
