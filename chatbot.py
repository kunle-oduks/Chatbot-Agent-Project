import pandas as pd
import numpy as np
import nltk
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random
lemmatizer = nltk.stem.WordNetLemmatizer()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

@st.cache_data()
def load_data(text, sep, header):
    data = pd.read_csv(text, sep = None, Header= None)
    return data

data = pd.read_csv('Telecoms.txt', sep = '?', header = None)

data.rename(columns = {0: 'Questions', 1: 'Answers'}, inplace = True)

def preprocessed_text(text):
    sentences = nltk.sent_tokenize(text)

    preprocessed_sentences = []
    for sentence in sentences:
        token = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        preprocessed_sentence = ' '.join(token)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)
   
data['tokenized'] = data['Questions'].apply(preprocessed_text)
data.head()

xtrain = data['tokenized'].to_list()

tfidfVectorizer = TfidfVectorizer()
corpus = tfidfVectorizer.fit_transform(xtrain)

#===========================================Streamlit Operations=========================================================

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-size: 60px; font-family: Helvetica'>WELCOME TO FIBERONYOU</h1>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Kunle Odukoya</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

space0, logo_image, space1, space2, space3 = st.columns(5)
with logo_image:
    st.image('pngwing.com (17).png', width = 600,  caption = "EMPOWERING CONNECTIVITY: EXPERIENCE THE SPEED OF LIGHT WITH OUR FIBER OPTIC SOLUTIONS")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

user_history = []
bot_history = []

missy_image, space1,space2, chats = st.columns(4)
with missy_image:
    missy_image.image('pngwing.com (15).png', width = 300)

with chats:
    user_message = chats.text_input('Hello, you can ask your question here: ')
    def responder(test):
        user_input_processed = preprocessed_text(test)
        vectorized_user_input = tfidfVectorizer.transform([user_input_processed])
        similarity_score = cosine_similarity(vectorized_user_input, corpus)
        argument_maximum = similarity_score.argmax()
        return(data['Response'].iloc[argument_maximum])

bot_greetings = ['Hello user, Welcome to FiberOnYou, please ask your question',
             'Hi, How are you today?  Welcome to FiberOnYou, How can I help you?',
             'Hallos, How do you do today? Welcome to FiberOnYou, is there anyway I can assist you?',
             'Welcome to FiberOnYou, How may I be of service to you?'
             ]

bot_farewell = ['Thanks for checking out FiberOnYou, I hope I have been of assistance,bye',
             'Hope, I have been helpful, bye',
             'Hope to see you soon and thanks for checking in',
             'Thanks for checking FiberOnYou, please do not hesiatate to contact us again',
             'I hope I have been of assistance, if you need further clarification, please visit again. Thanks'
           ]

human_greetings = ['hi', 'hello there', 'hey','hello']
human_exits = ['thanks bye', 'bye', 'quit','bye bye', 'close', 'exit']


random_greeting = random.choice(bot_greetings)
random_farewell = random.choice(bot_farewell)

if user_message.lower() in human_greetings:
    chats.write(f"\nChatbot: {random_greeting}")
    user_history.append(user_message)
    bot_history.append(random_greeting)

elif user_message.lower() in human_exits:
    chats.write(f"\nChatbot: {random_farewell}")
    user_history.append(user_message)
    bot_history.append(random_farewell)

elif  user_message == '':
    chats.write('')
else:
    response = responder(user_message)
    chats.write(f"\nChatbot: {response}")
    user_history.append(user_message)
    bot_history.append(response)


# Clearing Chat History 
def clearHistory():
    with open('user_history.txt', 'w') as file:
        pass  

    with open('bot_history.txt', 'w') as file:
        pass


#Saving chat history
import csv

with open('user_history.txt', 'a') as file:
    for item in user_history:
        file.write(str(item) + '\n')
    
with open('bot_history.txt', 'a') as file:
    for item in bot_history:
        file.write(str(item) + '\n')

with open('user_history.txt') as f:
    reader = csv.reader(f)
    data1 = list(reader)

with open('bot_history.txt') as f:
    reader = csv.reader(f)
    data2 = list(reader)

data1 = pd.Series(data1)
data2 = pd.Series(data2)

history = pd.DataFrame({"User Input": data1, "Agent's Response": data2})

st.subheader('Chat History', divider = True)
st.dataframe(history, use_container_width = True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button('End Chat'):
    clearHistory()
