import re
import gensim


def clean_text(text):
    # remove \n
    text = text.replace('\n', ' ')
    # remove punctuation
    text = re.sub(r'[,\.!?]', '', text)
    #removing text in square braquet
    text = re.sub(r'\[.*?\]', ' ', text)
    #removing numbers
    text = re.sub(r'\w*\d\w*',' ', text)
    #removing bracket
    text = re.sub(r'[()]', ' ', text)
    # convert all words in lower case
    text = text.lower()
    return text

def clean_lyrics(df):
    # get the results of data cleaning
    cleaned_text = df["lyrics"].apply(clean_text)
    # update dataframe
    df.update(cleaned_text)
    return df

# preprocess without ignoring stop words and lematize
def naive_preprocess(text, nlp):

    #TOKENISATION
    tokens =[]
    for token in nlp(text):
        tokens.append(token)
    
    # return list of words
    return [word.text for word in tokens if word.text.isalpha()]

    

# default preprocessing
def preprocess(text, nlp):

    #TOKENISATION
    tokens =[]
    for token in nlp(text):
        tokens.append(token)

    #REMOVING STOP WORDS
    spacy_stopwords = nlp.Defaults.stop_words
    sentence =  [word for word in tokens if word.text.isalpha() and word.text not in spacy_stopwords]

    #LEMMATISATION
    sentence = [word.lemma_ for word in sentence]

    return sentence


def ngram_models(df):
    # process in lyrics into words
    data = df['lyrics'].tolist()
    data_words = list((gensim.utils.simple_preprocess(str(sentence), deacc=True)
                       for sentence in data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_model = gensim.models.phrases.Phraser(bigram)
    trigram_model = gensim.models.phrases.Phraser(trigram)

    return bigram_model, trigram_model


# ngram preprocessing
def ngram_preprocess(text, nlp,
                     bigram_model,
                     trigram_model,
                     new_stopwords):

    # perform basic preprocessing to transform sentence to list of words
    words = gensim.utils.simple_preprocess(text)
    
    # customize stopwords
    spacy_stopwords = nlp.Defaults.stop_words
    ext_stopwords = spacy_stopwords | new_stopwords # union of set
    
    #removing stop words
    no_stop_words = [word for word in words if word not in ext_stopwords]
    
    # perform bigram model
    bigram_words = bigram_model[no_stop_words]
    
    # perform trigram model
    trigram_words = trigram_model[bigram_words]
    
    # recreate the sentence
    sentence = ' '.join(trigram_words)
    
    #tokenization to get lemma
    tokens = [token for token in nlp(sentence)]
    
    #LEMMATISATION and filter alphanumeric characters
    sentence = [word.lemma_ for word in tokens if word.text.isalpha()]

    return sentence