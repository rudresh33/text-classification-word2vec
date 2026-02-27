import re
import nltk
from sklearn.datasets import fetch_20newsgroups

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def load_and_preprocess_data():
    print("Loading 20 Newsgroups dataset...")
    # [cite_start]Load the 20 Newsgroups dataset using scikit-learn's fetch [cite: 37]
    # [cite_start]Remove headers, footers, and quoted replies from documents [cite: 38]
    remove_params = ('headers', 'footers', 'quotes')
    train_data = fetch_20newsgroups(subset='train', remove=remove_params)
    test_data = fetch_20newsgroups(subset='test', remove=remove_params)

    print(f"Loaded {len(train_data.data)} training documents and {len(test_data.data)} testing documents.")

    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        # [cite_start]Convert all text to lowercase [cite: 39]
        text = text.lower()
        
        # [cite_start]Remove punctuation marks using regular expressions [cite: 40]
        text = re.sub(r'[^\w\s]', '', text)
        
        # [cite_start]Tokenize text using whitespace splitting [cite: 42]
        tokens = text.split()
        
        # [cite_start]Remove English stopwords from NLTK stopword list [cite: 41]
        tokens = [word for word in tokens if word not in stop_words]
        
        return tokens

    print("Preprocessing training data...")
    train_texts_cleaned = [clean_text(doc) for doc in train_data.data]

    print("Preprocessing testing data...")
    test_texts_cleaned = [clean_text(doc) for doc in test_data.data]

    print("Preprocessing complete!")
    return train_texts_cleaned, train_data.target, test_texts_cleaned, test_data.target

if __name__ == "__main__":
    # Test the script
    train_texts, train_labels, test_texts, test_labels = load_and_preprocess_data()
    print(f"\nSample cleaned document (first 10 tokens): {train_texts[0][:10]}")
