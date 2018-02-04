from shared import *

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer

stemmer = SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

def main():
    df = pd.read_csv(TRAINING_FILE_LOC)
    
    toxic_df = df.loc[(df['toxic'] == 1) | (df['severe_toxic'] == 1) | (df['obscene'] == 1) | (df['threat'] == 1) | (df['insult'] == 1) | (df['identity_hate'] == 1)]

    """
    df['cleaned_comment_text'] = (df['comment_text']
                                        .map(lambda x: " ".join([stemmer.stem(i) for i in x.lower()
                                                              .split() if i not in stopwords.words("english")])))    
    """
    print toxic_df.shape
    comments = toxic_df['comment_text']
    
    vectorizer = StemmedCountVectorizer(analyzer='word', stop_words='english', min_df=0.0005, token_pattern="[a-z|A-Z]+")
    bag_words = vectorizer.fit_transform(comments).toarray()
    #print vectorizer.vocabulary_
    #print len(vectorizer.vocabulary_)
    #print vectorizer.get_feature_names()
    
    for comment in comments.values:
        blob = TextBlob(comment)
        polarity = [sentence.sentiment.polarity for sentence in blob.sentences]
        continue
    return

if __name__ == "__main__":
    main()