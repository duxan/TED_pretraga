import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, SelectField

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '12345fffff'

class ReusableForm(Form):
    keywords = TextField('keywords:', validators=[validators.DataRequired()])
    
    @app.route("/", methods=['GET', 'POST'])
    def hello():
        form = ReusableForm(request.form)
        
        if request.method == 'POST':
            keywords=request.form['keywords']
            model=request.form['model']
        
        urls = []
        if form.validate():
            # TODO: transform and clean input keywords in the same way as docs
            if model == "cleaned":
                vectorizer = vectorizer_cleaned
                doc_weights = doc_weights_cleaned
                # keywords = 
            elif model == "no_stemm-stop":
                vectorizer = vectorizer_no_stemm_stop
                doc_weights = doc_weights_no_stemm_stop
            elif model == "stemm-no_stop":
                vectorizer = vectorizer_stemm_no_stop
                doc_weights = doc_weights_stemm_no_stop
            elif model == "stemm-stop":
                vectorizer = vectorizer_stemm_stop
                doc_weights = doc_weights_stemm_stop
                
            search_weights = query_weights(vectorizer, [keywords])
            idx = most_similar_idx(cos_similarity(search_weights, doc_weights), min_talks=3)
            urls = ["https://embed.ted.com/talks/" + url.split("talks/")[1] for url in most_similar_urls(df, idx)]
            urls = [url.replace("/transcript", "") for url in urls]
            flash('Found videos')
        else:
            flash('Error: All the form fields are required. ')
        
        return render_template('form.html', form=form, urls=urls)


def tf_idf(dataframe, label):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1)) # TODO: check other options
    vectorizer = tfidf_vectorizer.fit(dataframe.loc[:, label])
    tfidf_weights_matrix = vectorizer.transform(dataframe.loc[:, label])

    return tfidf_weights_matrix, vectorizer


def query_weights(vectorizer, list_of_queries):
    return vectorizer.transform(list_of_queries)


def cos_similarity(search_query_weights, tfidf_weights_matrix):
    similarities = cosine_similarity(search_query_weights, tfidf_weights_matrix)
    return similarities[0]
  

def most_similar_idx(similarity_list, min_talks=3):
    most_similar= []
  
    while min_talks > 0:
        tmp_index = np.argmax(similarity_list)
        most_similar.append(tmp_index)
        similarity_list[tmp_index] = 0
        min_talks -= 1

    return most_similar


def most_similar_urls(df, idx):
    return df.loc[idx, "link"].to_list()


if __name__ == "__main__":
    df = pd.read_csv("data_processed.csv")
    doc_weights_cleaned, vectorizer_cleaned = tf_idf(df, "cleaned")
    doc_weights_no_stemm_stop, vectorizer_no_stemm_stop = tf_idf(df, "no_stemm-stop")
    doc_weights_stemm_stop, vectorizer_stemm_stop = tf_idf(df, "stemm-stop")
    doc_weights_stemm_no_stop, vectorizer_stemm_no_stop = tf_idf(df, "stemm-no_stop")
    
    app.run(port=4001)
