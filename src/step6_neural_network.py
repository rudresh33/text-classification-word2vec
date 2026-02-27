import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score
from step4_custom_word2vec import train_and_extract_vectors

def train_evaluate_fnn():
    print("--- Starting Neural Network Pipeline ---")
    
    # Run the previous pipeline to get the dense vectors
    X_train, y_train, X_test, y_test, _ = train_and_extract_vectors()

    # The loss function 'categorical_crossentropy' requires one-hot encoded labels
    y_train_encoded = to_categorical(y_train, num_classes=20)
    y_test_encoded = to_categorical(y_test, num_classes=20)

    # Step 6: Proposed Model Architecture
    print("\nBuilding Shallow Feed-Forward Neural Network (FNN)...")
    model = Sequential([
        Input(shape=(100,)),                           # Input Layer: 100 neurons [cite: 67]
        Dropout(0.5),                                  # Dropout Layer: 0.5 dropout rate [cite: 68]
        Dense(128, activation='relu'),                 # Hidden Layer 1: 128 neurons, ReLU [cite: 69]
        Dense(64, activation='relu'),                  # Hidden Layer 2: 64 neurons, ReLU [cite: 70]
        Dense(20, activation='softmax')                # Output Layer: 20 neurons, Softmax [cite: 71]
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.001)              # Optimizer: Adam with learning_rate=0.001 [cite: 72]
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',     # Loss function: Categorical cross-entropy [cite: 73]
                  metrics=['accuracy'])

    model.summary()

    # Early stopping configuration
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5,                                    # early stopping (patience=5) [cite: 74]
        restore_best_weights=True
    )

    print("\nTraining FNN...")
    history = model.fit(
        X_train, y_train_encoded,
        epochs=50,                                     # Training epochs: 50 [cite: 74]
        batch_size=32,                                 # Batch size: 32 [cite: 73]
        validation_split=0.1,                          # Validation split: 10% of training data [cite: 74]
        callbacks=[early_stopping],
        verbose=1
    )

    # Step 7: Experiment and Comparison 
    print("\nEvaluating Proposed Model on Test Set...")
    # Get raw predictions and convert back to class indices for F1-score calculation
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compare Accuracy and macro-averaged F-Score [cite: 77]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print("\n" + "="*35)
    print("   PROPOSED METHOD RESULTS (FNN)")
    print("="*35)
    print(f"Accuracy:                {accuracy:.4f}")
    print(f"Macro-Averaged F1-Score: {f1:.4f}")
    print("="*35)
    
    return accuracy, f1

if __name__ == "__main__":
    # Ensure TensorFlow doesn't output excessive warning logs
    tf.get_logger().setLevel('ERROR')
    train_evaluate_fnn()
