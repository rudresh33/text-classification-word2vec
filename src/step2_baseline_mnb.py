from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from step1_preprocessing import load_and_preprocess_data

def run_mnb_baseline():
    print("--- Starting Traditional Pipeline: TF-IDF + MNB ---")
    
    # 1. Load data from physical directories
    train_texts, train_labels, test_texts, test_labels, target_names = load_and_preprocess_data('full')

    print("\nRejoining tokens into strings for TF-IDF...")
    train_strings = [" ".join(tokens) for tokens in train_texts]
    test_strings = [" ".join(tokens) for tokens in test_texts]

    # 2. Extract Features
    print("Extracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, min_df=2, max_df=0.95)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_strings)
    X_test_tfidf = tfidf_vectorizer.transform(test_strings)

    # 3. Train Model
    print("Training Multinomial Naive Bayes model...")
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train_tfidf, train_labels)

    # 4. Evaluate
    y_pred = mnb.predict(X_test_tfidf)
    accuracy = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, average='macro')
    
    print("\n" + "="*35)
    print(" METHOD 2: TF-IDF + NAIVE BAYES")
    print("="*35)
    print(f"Accuracy:                {accuracy:.4f}")
    print(f"Macro-Averaged F1-Score: {f1:.4f}")
    print("="*35)
    
    return accuracy, f1

if __name__ == "__main__":
    run_mnb_baseline()
