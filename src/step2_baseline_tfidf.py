from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from step1_preprocessing import load_and_preprocess_data

def run_baseline():
    print("--- Starting Baseline Pipeline ---")
    # Step 1: Get preprocessed data
    train_texts, train_labels, test_texts, test_labels = load_and_preprocess_data()

    # TF-IDF vectorizer expects strings, not lists of tokens, so we join them
    print("\nRejoining tokens into strings for TF-IDF...")
    train_strings = [" ".join(tokens) for tokens in train_texts]
    test_strings = [" ".join(tokens) for tokens in test_texts]

    # Step 2: Baseline Feature Extraction (TF-IDF)
    print("Extracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.95
    )
    
    # Fit on training data and transform both sets
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_strings)
    X_test_tfidf = tfidf_vectorizer.transform(test_strings)
    
    print(f"TF-IDF training feature matrix shape: {X_train_tfidf.shape}")

    # Step 3: Baseline Model Training (Multinomial Naive Bayes)
    print("\nTraining Multinomial Naive Bayes model...")
    mnb_classifier = MultinomialNB(alpha=1.0)
    mnb_classifier.fit(X_train_tfidf, train_labels)

    # Evaluation
    print("Evaluating baseline model on test set...")
    y_pred = mnb_classifier.predict(X_test_tfidf)
    
    accuracy = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, average='macro')
    
    print("\n" + "="*30)
    print("      BASELINE RESULTS")
    print("="*30)
    print(f"Accuracy:                {accuracy:.4f}")
    print(f"Macro-Averaged F1-Score: {f1:.4f}")
    print("="*30)
    
    return accuracy, f1

if __name__ == "__main__":
    run_baseline()
