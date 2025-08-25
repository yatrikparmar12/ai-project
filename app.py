#========================import packages=========================================================
import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk
import pandas as pd
import altair as alt

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

#========================loading the saved files==================================================
lg = pickle.load(open('logistic_regresion.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))

# =========================text preprocessing function==========================================
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# =========================emotion prediction function==========================================
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    
    # Get prediction probabilities
    prediction_probabilities = lg.predict_proba(input_vectorized)[0]
    
    return predicted_emotion, prediction_probabilities


#=========================Mapping emotions to emojis============================================
emotions_emoji_dict = {
    "anger": "😠", "fear": "😨😱",  "joy": "😂", "sadness": "😔",  "surprise": "😮", "love": "❤️"
}

#==================================Creating Streamlit App====================================
def main():
    st.set_page_config(page_title="Emotion Prediction App", page_icon=":sunglasses:")
    st.title("HUMAN EMOTION PREDICTION APP")
    st.write(["Joy","Fear","Love","Anger","Sadness","Surprise"])
    st.write("This app can predict emotions from text input.")
    
    input_text = st.text_input("ENTER YOUR TEXT HERE:")

    if st.button("PREDICT"):
        if input_text.strip() == "":
            st.warning("PLEASE ENTER SOME TEXT TO PREDICT THE EMOTION.")
        else:
            # Predict emotion
            predicted_emotion, probabilities = predict_emotion(input_text)

            # Display prediction results
            st.success("PREDICTED EMOTION:")
            st.write(f"**{predicted_emotion.upper()}** {emotions_emoji_dict.get(predicted_emotion, '❓')}")

            # **Fix for np.max() error**
            st.write("CONFIDENCE: {:.2f}".format(np.max(probabilities)))

            st.success("PREICTION PROBABILITIES:")
            emotion_labels = lb.classes_
            proba_df = pd.DataFrame({"Emotion": emotion_labels, "Probability": probabilities})
            
            # Create a bar chart
            fig = alt.Chart(proba_df).mark_bar().encode(
                x=alt.X('Emotion', sort='-y'),
                y='Probability',
                color='Emotion'
                )

            st.altair_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()  
