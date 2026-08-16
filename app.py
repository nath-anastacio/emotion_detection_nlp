import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Downloading NLTK's necessary resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# --- Loading model, vectorizer and label encoder

@st.cache_resource
# @st.cache_resource avoids reloading the model everytime the user interacts with the interface (improves the demo's performance)
def load_artifacts():
    model = joblib.load('model/emotion_svm_model.pkl')
    tfidf = joblib.load('model/tfidf_vectorizer.pkl')
    le = joblib.load('model/label_encoder.pkl')
    return model, tfidf, le

model, tfidf, le = load_artifacts()

stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# The app needs the same function used during training, so it's rewritten here (the app does not have access to the notebook)

def get_wordnet_pos(tag):
    if tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)        # Remove mentions/hashtags
    text = re.sub(r'[^a-z\s]', '', text)         # Remove punctiations/numbers

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags
    ]

    tokens_final = [word for word in lemmatized if word not in stopwords]
    return ' '.join(tokens_final)

# --- Interface ---

st.set_page_config(page_title='Emotion Detection')

st.title('Emotion Detection 🎭')
st.write('Write a text in english and see what emotion is detected by the model:')

user_input = st.text_area('Your text:', placeholder="Ex: 'I am so happy today, everything feels amazing'")

if st.button('Detect Emotion'):
    if user_input.strip() == '':
        st.warning('Please, write a text.')
    else:
        cleanned = clean_text(user_input)
        vector = tfidf.transform([cleanned])
        prediction = model.predict(vector)
        emotion = le.inverse_transform(prediction)[0]

        emotion_map = {
            'joy':'🙂' , 'sadness' : '🙁' , 'anger' : '😠' , 'fear' : '😨' , 'love' : '❤️' , 'surprise' : '😱'
        }

        st.subheader(f"Detected emotion: {emotion.upper()} {emotion_map.get(emotion, '')}")

        with st.expander('See details'):
            st.write(f"Original text: {user_input}")
            st.write(f"Cleanned up text: {cleanned}")


st.markdown("---")
st.caption('Model: Linear SVM + TF-IDF | Test Accuracy: 86% | F1 macro: 0.83')