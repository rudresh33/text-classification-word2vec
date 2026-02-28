import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups

def create_local_dataset(bunch, subset_name, dataset_type, max_docs_per_class=None):
    """Writes text documents to physical hierarchical folders on the hard drive."""
    print(f"Writing {dataset_type} dataset ({subset_name} split)...")
    target_names = bunch.target_names
    class_counts = {name: 0 for name in target_names}
    
    base_dir = f"data/{dataset_type}/{subset_name}"
    os.makedirs(base_dir, exist_ok=True)

    for i, (text, target_idx) in enumerate(zip(bunch.data, bunch.target)):
        category = target_names[target_idx]
        
        # Limit documents if building the <10MB sample zip
        if max_docs_per_class and class_counts[category] >= max_docs_per_class:
            continue
            
        cat_dir = os.path.join(base_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        # Write to physical text file
        file_path = os.path.join(cat_dir, f"doc_{i}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        class_counts[category] += 1

def generate_eda_plots(train_bunch, test_bunch):
    """Generates Exploratory Data Analysis visualizations for the report/PPT."""
    print("\nGenerating EDA visualizations...")
    os.makedirs('results/eda', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    data_records = []
    
    # Process Train and Test lengths
    for text, target_idx in zip(train_bunch.data, train_bunch.target):
        words = len(text.split())
        if words > 0:
            data_records.append({'Category': train_bunch.target_names[target_idx], 'Word Count': words, 'Split': 'Train'})
            
    for text, target_idx in zip(test_bunch.data, test_bunch.target):
        words = len(text.split())
        if words > 0:
            data_records.append({'Category': test_bunch.target_names[target_idx], 'Word Count': words, 'Split': 'Test'})

    df = pd.DataFrame(data_records)

    # Plot 1: Length Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Word Count', hue='Split', bins=50, log_scale=True, element="step", common_norm=False)
    plt.title('Distribution of Document Word Counts (Full Dataset)')
    plt.xlabel('Number of Words (Log Scale)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('results/eda/word_count_distribution.png', dpi=300)
    plt.close()

    # Plot 2: Class Balance
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, y='Category', hue='Split', order=df['Category'].value_counts().index, palette="viridis")
    plt.title('Number of Documents per Category (Full Dataset)')
    plt.xlabel('Number of Documents')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig('results/eda/class_distribution.png', dpi=300)
    plt.close()
    
    print("Saved EDA plots to 'results/eda/'")

def zip_submission_dataset():
    """Compresses the sample data directory for the Google Form."""
    zip_name = 'dataset.zip'
    print(f"\nCompressing 'data/sample' into '{zip_name}'...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk('data/sample'):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, 'data/sample')
                zipf.write(file_path, arcname)
    
    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"Success! '{zip_name}' created. Final Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    print("--- Extracting Raw Text to Physical Drives & Generating EDA ---")
    remove_params = ('headers', 'footers', 'quotes')
    train_bunch = fetch_20newsgroups(subset='train', remove=remove_params)
    test_bunch = fetch_20newsgroups(subset='test', remove=remove_params)
    
    create_local_dataset(train_bunch, 'train', 'full')
    create_local_dataset(test_bunch, 'test', 'full')
    
    create_local_dataset(train_bunch, 'train', 'sample', max_docs_per_class=40)
    create_local_dataset(test_bunch, 'test', 'sample', max_docs_per_class=20)
    
    generate_eda_plots(train_bunch, test_bunch)
    zip_submission_dataset()
    print("--- Physical Datasets & EDA Created Successfully ---")
