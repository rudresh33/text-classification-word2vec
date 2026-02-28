import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score
from step1_preprocessing import load_and_preprocess_data

def get_document_vector(tokens, model):
    """Averages word vectors to create a single document vector."""
    valid_words = [word for word in tokens if word in model.wv]
    if valid_words:
        return np.mean(model.wv[valid_words], axis=0)
    else:
        return np.zeros(model.vector_size)

def run_fnn_pipeline():
    print("--- Starting Deep Learning Pipeline: Word2Vec + FNN ---")
    
    # 1. Load data from physical directories
    train_texts, train_labels, test_texts, test_labels, _ = load_and_preprocess_data('full')

    # 2. Train Word2Vec
    print("\nTraining custom Word2Vec model...")
    w2v_model = Word2Vec(
        sentences=train_texts, vector_size=100, window=5, 
        min_count=2, sg=1, epochs=10, workers=4
    )

    # 3. Extract Document Vectors
    print("Extracting averaged Word2Vec document vectors...")
    X_train_dense = np.array([get_document_vector(doc, w2v_model) for doc in train_texts])
    X_test_dense = np.array([get_document_vector(doc, w2v_model) for doc in test_texts])

    y_train_encoded = to_categorical(train_labels, num_classes=20)
    y_test_encoded = to_categorical(test_labels, num_classes=20)

    # 4. Build FNN Architecture
    print("\nBuilding Shallow Feed-Forward Neural Network (FNN)...")
    model = Sequential([
        Input(shape=(100,)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(20, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 5. Train FNN
    print("Training FNN...")
    model.fit(
        X_train_dense, y_train_encoded, 
        epochs=50, batch_size=32, validation_split=0.1, 
        callbacks=[early_stopping], verbose=1
    )

    # 6. Evaluate
    print("\nEvaluating Proposed Model on Test Set...")
    y_pred_probs = model.predict(X_test_dense)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, average='macro')

    print("\n" + "="*35)
    print("   METHOD 4: WORD2VEC + FNN")
    print("="*35)
    print(f"Accuracy:                {accuracy:.4f}")
    print(f"Macro-Averaged F1-Score: {f1:.4f}")
    print("="*35)

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    run_fnn_pipeline()
