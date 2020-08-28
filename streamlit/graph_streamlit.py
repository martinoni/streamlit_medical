import srsly
import codecs
import streamlit as st
import spacy_streamlit
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from spacy_streamlit import load_model
import streamlit.components.v1 as stc
from nltk.tokenize import sent_tokenize

def st_graph(html):
    graph_file = codecs.open(html, 'r')
    page = graph_file.read()
    stc.html(page, width=800, height=600)

def pre_processing(text):
    text = text.replace('–','')
    text = text.replace('…',' ')
    text = text.replace('[','')
    text = text.replace(']','')
    text = text.replace('’s','')
    text = text.replace('’','')
    text = text.strip()
    return text

def page_rank(n, page_graph):
    ranks = nx.pagerank(page_graph, alpha=0.9)
    summary = sorted(ranks.keys(), key=lambda k: ranks[k], reverse=True)[:n]
    return summary, ranks

graph_file = srsly.read_json('graph_json.json')

#Pre-Processing 
transcription = srsly.read_json('medical_transcription.json')
transcription_cleaned = pre_processing(transcription['text'])
sentences = sent_tokenize(transcription_cleaned)

st.sidebar.image('logo_cdr.png', use_column_width=True)
#Load Model
st.sidebar.subheader('Model')
model = st.sidebar.selectbox("Model name", ["Clinical", "General"])

if model == 'Clinical':
    nlp = load_model("en_ner_bc5cdr_md")
elif model == 'General':
    nlp = load_model("en_core_web_sm")

#Graph
st.sidebar.subheader('Graph')
threshold = st.sidebar.slider("Similarity Threshold",0.0, 1.0)
st.title('Medical Graph')

page_graph = nx.Graph()
graph = Network(width='100%',bgcolor="white", font_color="#444444", directed=True, heading='', notebook=True)
for link in graph_file['links']:
    similarity = float(link['weight'])
    if similarity >= threshold and similarity != 1.0:
        graph.add_node(link['source'], size=14, color = '#2169AD')
        graph.add_node(link['target'], size=10, color = '#78CABC')
        graph.add_edge(link['source'], link['target'], weight = round(similarity * 100))
        page_graph.add_edge(link['source'], link['target'], weight = similarity)

graph.force_atlas_2based()
graph.show('graph.html')
st_graph('graph.html')

#Page Rank
st.header('Page Rank')
n_words = st.sidebar.slider("Page Rank", 1, 20, 10)
summary, ranks = page_rank(n_words, page_graph)
word_dict = {word: ranks[word] for word in summary[:n_words]}
plt.barh(list(word_dict.keys()), list(word_dict.values()), align='center')
plt.gca().invert_yaxis()
st.pyplot()

#NER POS
st.subheader("Sentence")
text = st.selectbox("", sentences)
doc = nlp(text)
spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, show_table=False, title='Named Entity Recognition')
spacy_streamlit.visualize_parser(doc)
