from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from step1_preprocessing import load_and_preprocess_data

def run_proposed_solution():
    print("--- Starting Proposed Solution: Pre-trained Transformers + SVM ---")
    
    # 1. Load data
    train_texts, train_labels, test_texts, test_labels, target_names = load_and_preprocess_data('full')
    train_strings = [" ".join(tokens) for tokens in train_texts]
    test_strings = [" ".join(tokens) for tokens in test_texts]

    # 2. Encode text using lightweight pre-trained contextual embeddings
    print("\nEncoding documents using all-MiniLM-L6-v2 (this may take a few minutes)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    X_train_dense = model.encode(train_strings, show_progress_bar=True)
    X_test_dense = model.encode(test_strings, show_progress_bar=True)

    # 3. Train SVM on the contextual dense embeddings
    print("\nTraining Linear Support Vector Classifier on Transformer Embeddings...")
    svm_model = LinearSVC(random_state=42, dual=False, max_iter=2000) 
    svm_model.fit(X_train_dense, train_labels)

    # 4. Evaluate
    print("Evaluating Proposed Solution...")
    y_pred = svm_model.predict(X_test_dense)
    accuracy = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, average='macro')
    
    print("\n" + "="*45)
    print(" METHOD 7: SENTENCE TRANSFORMERS + SVM (PROPOSED)")
    print("="*45)
    print(f"Accuracy:                {accuracy:.4f}")
    print(f"Macro-Averaged F1-Score: {f1:.4f}")
    print("="*45)

if __name__ == "__main__":
    run_proposed_solution()
