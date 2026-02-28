import os
import re
import nltk
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def read_local_documents(base_path):
    """Crawls physical directories to read text files and their category labels."""
    texts = []
    labels = []
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Directory not found: {base_path}. Run step0 first!")
        
    for category in sorted(os.listdir(base_path)):
        category_path = os.path.join(base_path, category)
        
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(category)
                
    return texts, labels

def clean_text(text, stop_words):
    """Applies traditional NLP text cleaning."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def load_and_preprocess_data(dataset_type='full'):
    """Main pipeline to load physical files, clean them, and encode labels."""
    print(f"Reading physical files from 'data/{dataset_type}/'...")
    
    train_dir = f"data/{dataset_type}/train"
    test_dir = f"data/{dataset_type}/test"
    
    train_raw_texts, train_raw_labels = read_local_documents(train_dir)
    test_raw_texts, test_raw_labels = read_local_documents(test_dir)
    
    print(f"Loaded {len(train_raw_texts)} training files and {len(test_raw_texts)} testing files.")
    
    # Encode folder names (categories) into numbers (0-19)
    le = LabelEncoder()
    train_labels = le.fit_transform(train_raw_labels)
    test_labels = le.transform(test_raw_labels)
    
    stop_words = set(stopwords.words('english'))
    
    print("Cleaning and tokenizing training data...")
    train_texts_cleaned = [clean_text(doc, stop_words) for doc in train_raw_texts]
    
    print("Cleaning and tokenizing testing data...")
    test_texts_cleaned = [clean_text(doc, stop_words) for doc in test_raw_texts]
    
    print("Preprocessing complete!")
    return train_texts_cleaned, train_labels, test_texts_cleaned, test_labels, le.classes_

if __name__ == "__main__":
    train_texts, train_labels, test_texts, test_labels, target_names = load_and_preprocess_data()
    print(f"\nCategories found: {len(target_names)}")
    print(f"Sample cleaned document (first 10 tokens): {train_texts[0][:10]}")
    print(f"Assigned label index: {train_labels[0]} (Category: {target_names[train_labels[0]]})")
