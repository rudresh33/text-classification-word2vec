import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from step1_preprocessing import load_and_preprocess_data

def plot_model_comparison():
    """Generates a bar chart comparing all models from the ablation study."""
    print("Generating Model Comparison Plot...")
    
    # Data from your final terminal runs
    data = {
        'Method': [
            'M2: TF-IDF + MNB', 
            'M3: TF-IDF + SVM', 
            'M4: Word2Vec + FNN', 
            'M5: 1D CNN', 
            'M6: Fixed LSTM', 
            'M7: S-Transformers + SVM\n(Proposed)'
        ],
        'Accuracy': [66.28, 65.80, 53.61, 39.87, 6.84, 66.53],
        'F1-Score': [63.42, 64.70, 49.37, 34.17, 2.64, 65.03]
    }
    
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='Method', var_name='Metric', value_name='Score (%)')

    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    # Create the grouped bar chart
    ax = sns.barplot(x='Method', y='Score (%)', hue='Metric', data=df_melted, palette=['#4C72B0', '#55A868'])
    
    plt.title('Ablation Study Results: Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Score (%)', fontsize=12)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylim(0, 80)
    plt.xticks(rotation=15)
    
    # Add the exact numbers on top of the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300)
    plt.close()
    print("Saved -> results/model_comparison.png")

def plot_winning_confusion_matrix():
    """Trains Method 7 quickly and plots its confusion matrix to show misclassifications."""
    print("\nGenerating Confusion Matrix for Method 7 (This will take a minute to encode text)...")
    
    # Load data
    train_texts, train_labels, test_texts, test_labels, target_names = load_and_preprocess_data('full')
    train_strings = [" ".join(tokens) for tokens in train_texts]
    test_strings = [" ".join(tokens) for tokens in test_texts]

    # Encode and Train (Method 7)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X_train_dense = model.encode(train_strings, show_progress_bar=False)
    X_test_dense = model.encode(test_strings, show_progress_bar=False)
    
    svm_model = LinearSVC(random_state=42, dual=False, max_iter=2000) 
    svm_model.fit(X_train_dense, train_labels)
    y_pred = svm_model.predict(X_test_dense)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(test_labels, y_pred)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix: Method 7 (Sentence Transformers + SVM)', fontsize=16, pad=20)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.ylabel('True Category', fontsize=12)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/method7_confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved -> results/method7_confusion_matrix.png")

if __name__ == "__main__":
    plot_model_comparison()
    plot_winning_confusion_matrix()
