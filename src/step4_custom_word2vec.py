import numpy as np
from gensim.models import Word2Vec
from step1_preprocessing import load_and_preprocess_data

def train_and_extract_vectors():
    print("--- Starting Word2Vec Pipeline ---")
    # Step 1: Get preprocessed data
    train_texts, train_labels, test_texts, test_labels = load_and_preprocess_data()

    # Step 4: Proposed Feature Training (Custom Word2Vec)
    print("\nTraining custom Word2Vec model on the training corpus...")
    # Configuration based on project requirements:
    # Skip-gram (sg=1), vector size=100, window=5, min_count=2, epochs=10, negative sampling=5
    w2v_model = Word2Vec(
        sentences=train_texts,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1, 
        negative=5,
        epochs=10,
        workers=4 # Utilizing multiple cores to speed up training
    )
    print(f"Word2Vec vocabulary built with {len(w2v_model.wv)} words.")

    # Step 5: Proposed Feature Extraction (Vector Averaging)
    def get_document_vector(tokens, model):
        # Retrieve 100-dimensional Word2Vec vectors for all tokens [cite: 59]
        # Handle out-of-vocabulary words by skipping them 
        valid_words = [word for word in tokens if word in model.wv]
        
        if valid_words:
            # Create document vectors by taking the element-wise mean [cite: 60]
            return np.mean(model.wv[valid_words], axis=0)
        else:
            # Fallback for empty documents after filtering to maintain 100 dimensions [cite: 62]
            return np.zeros(model.vector_size)

    print("\nExtracting averaged Word2Vec document vectors...")
    X_train_dense = np.array([get_document_vector(doc, w2v_model) for doc in train_texts])
    X_test_dense = np.array([get_document_vector(doc, w2v_model) for doc in test_texts])

    print(f"Dense training feature matrix shape: {X_train_dense.shape}")
    print(f"Dense testing feature matrix shape: {X_test_dense.shape}")

    return X_train_dense, train_labels, X_test_dense, test_labels, w2v_model

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, model = train_and_extract_vectors()
    print("\nSample document vector (first 5 dimensions):")
    print(X_train[0][:5])
