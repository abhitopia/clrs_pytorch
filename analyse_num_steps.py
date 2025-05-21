from collections import defaultdict
from clrs.algorithm import Algorithm, CLRS30Algorithms, AlgorithmEnum
from clrs.dataset_archive import SIZES_MAX_NUM_STEPS, DEFAULT_SIZES, DEFAULT_MAX_NUM_STEPS
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os



def main():
    # sizes = [4, 7, 11, 13, 16]
    num_samples = 1000
    show_outliers = True  # Set to False if you don't want to see individual outlier points

    num_steps = {}

    algos = CLRS30Algorithms
    progress_bar = tqdm(algos)
    for algo in progress_bar:
        progress_bar.set_description(f"Processing {algo}")
        num_steps[algo] = defaultdict(list)

        sizes = SIZES_MAX_NUM_STEPS[algo][0] if algo in SIZES_MAX_NUM_STEPS else DEFAULT_SIZES
        for size in sizes:
            sampler = Algorithm(algo, length=size)
            for _ in range(num_samples):
                feature = sampler.sample_feature()
                num_steps[algo][size].append(feature[1])

    # Convert data to DataFrame for easier plotting
    data = []
    for algo in num_steps:
        for size in num_steps[algo]:
            for steps in num_steps[algo][size]:
                data.append({
                    'Algorithm': algo.name,
                    'Size': size,
                    'Steps': steps
                })
    df = pd.DataFrame(data)

    # Set the style
    plt.style.use('seaborn-v0_8')
    
    # Create output directory if it doesn't exist
    os.makedirs('algorithm_plots', exist_ok=True)
    
    # Create separate plots for each algorithm
    for algo_name in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo_name]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Box plot
        sns.boxplot(data=algo_data, x='Size', y='Steps', ax=ax1, 
                    showfliers=show_outliers,
                    fliersize=3)
        ax1.set_title(f'Distribution of Steps for {algo_name}', pad=20, fontsize=12)
        ax1.set_xlabel('Input Size', labelpad=10)
        ax1.set_ylabel('Number of Steps', labelpad=10)
        
        # Add a text box explaining the box plot
        if show_outliers:
            textstr = 'Box plot explanation:\n' + \
                     '• Box shows 25th to 75th percentile\n' + \
                     '• Line in box is median\n' + \
                     '• Whiskers extend to 1.5×IQR\n' + \
                     '• Dots show individual outliers'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=8,
                    verticalalignment='top', bbox=props)
        
        # Plot 2: Mean and 90th percentile
        stats = algo_data.groupby('Size').agg({
            'Steps': ['mean', lambda x: np.percentile(x, 90)]
        }).reset_index()
        stats.columns = ['Size', 'Mean', '90th Percentile']
        
        # Melt the dataframe for easier plotting
        stats_melted = pd.melt(stats, 
                              id_vars=['Size'],
                              value_vars=['Mean', '90th Percentile'],
                              var_name='Statistic',
                              value_name='Steps')
        
        sns.lineplot(data=stats_melted, x='Size', y='Steps', 
                    style='Statistic', marker='o', ax=ax2)
        ax2.set_title(f'Mean and 90th Percentile Steps for {algo_name}', pad=20, fontsize=12)
        ax2.set_xlabel('Input Size', labelpad=10)
        ax2.set_ylabel('Number of Steps', labelpad=10)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'algorithm_plots/{algo_name}_analysis.png', bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    main()