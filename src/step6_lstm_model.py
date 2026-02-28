import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score
from step1_preprocessing import load_and_preprocess_data

def run_fixed_lstm():
    print("--- Starting Deep Learning Pipeline: Fixed LSTM with Pre-trained Weights ---")
    
    # 1. Load data
    train_texts, train_labels, test_texts, test_labels, _ = load_and_preprocess_data('full')
    train_strings = [" ".join(tokens) for tokens in train_texts]
    test_strings = [" ".join(tokens) for tokens in test_texts]

    # 2. Train quick Word2Vec to get weights for the Embedding layer
    print("\nTraining Word2Vec for Embedding Initialization...")
    w2v_model = Word2Vec(sentences=train_texts, vector_size=100, window=5, min_count=2, sg=1, epochs=10, workers=4)

    # 3. Tokenize and Pad Sequences (Reduced max_len to 200 to prevent gradient vanishing)
    print("\nTokenizing and padding sequences for LSTM...")
    max_vocab = 20000
    max_len = 200  
    
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(train_strings)
    
    X_train_seq = tokenizer.texts_to_sequences(train_strings)
    X_test_seq = tokenizer.texts_to_sequences(test_strings)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
    
    y_train_encoded = to_categorical(train_labels, num_classes=20)

    # 4. Create Embedding Matrix
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((max_vocab, 100))
    for word, i in word_index.items():
        if i < max_vocab and word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    # 5. Build LSTM Architecture
    print("Building LSTM Architecture...")
    model = Sequential([
        Embedding(input_dim=max_vocab, output_dim=100, weights=[embedding_matrix], input_length=max_len, trainable=False),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(20, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 6. Train LSTM
    print("Training LSTM...")
    model.fit(X_train_pad, y_train_encoded, epochs=15, batch_size=64, validation_split=0.1, callbacks=[early_stopping], verbose=1)

    # 7. Evaluate
    print("\nEvaluating Fixed LSTM Model on Test Set...")
    y_pred_probs = model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, average='macro')

    print("\n" + "="*35)
    print("       METHOD 6: FIXED LSTM")
    print("="*35)
    print(f"Accuracy:                {accuracy:.4f}")
    print(f"Macro-Averaged F1-Score: {f1:.4f}")
    print("="*35)

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    run_fixed_lstm()
