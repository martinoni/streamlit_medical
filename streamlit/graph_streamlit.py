from srsly import read_json
from codecs import open as codecs_open
import streamlit as st
from networkx import pagerank, Graph
import matplotlib.pyplot as plt
from pyvis.network import Network
from spacy_streamlit import load_model, visualize_ner, visualize_parser
import streamlit.components.v1 as stc
from nltk.tokenize import sent_tokenize
from numpy import array
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from plotly.express import scatter_3d


def st_graph(html):
    graph_file = codecs_open(html, 'r')
    page = graph_file.read()
    stc.html(page, width=1000, height=600)

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
    ranks = pagerank(page_graph, alpha=0.6)
    summary = sorted(ranks.keys(), key=lambda k: ranks[k], reverse=True)[:n]
    return summary, ranks

graph_file = read_json('graph_json.json')

#Pre-Processing 
transcription = read_json('medical_transcription.json')
transcription_cleaned = pre_processing(transcription['text'])
sentences = sent_tokenize(transcription_cleaned)

st.sidebar.image('logo_cdr.png', use_column_width=True)
#Load Model
st.sidebar.subheader('Model')
model = st.sidebar.selectbox("Model name", ["Clinical", "General"])

if model == 'Clinical':
    nlp = load_model("en_ner_bc5cdr_md/en_ner_bc5cdr_md/en_ner_bc5cdr_md-0.2.5")
elif model == 'General':
    nlp = load_model("en_core_web_sm")

#Graph
st.sidebar.subheader('Graph')
threshold = st.sidebar.slider("Similarity Threshold",0.0, 1.0)
st.title('Medical Graph')

remove_stop = st.checkbox('Remove stopwords')
if remove_stop:
    graph_file = read_json('graph_no_stop.json')
else:
    graph_file = read_json('graph_with_stop.json')

page_graph = Graph()
graph = Network(width='100%',bgcolor="white", font_color="#444444", directed=True, heading='', notebook=True)
for link in graph_file['links']:
    similarity = float(link['weight'])
    if similarity >= threshold and similarity != 1.0:
        graph.add_node(link['source'], size=10, color = '#78CABC')
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
visualize_ner(doc, labels=nlp.get_pipe("ner").labels, show_table=False, title='Named Entity Recognition')
visualize_parser(doc)

#Umap 3D
st.subheader("Embedding Visualization")
words = ['weakness', 'suddenly', 'strength', 'laughter', 'cause', 'dementia', 'scar', 'exam', 'moment', 'slowly', 'contract', 'disease', 'appearance', 'rest', 'my', 'dreadful', 'concern', 'rush', 'doctor', 'husband', 'maybe', 'problem', 'part', 'pull', 'compensate', 'concerned', 'heal', 'function', 'probably', 'hate', 'something', 'there', 'person', 'specialise', 'particularly', 'work', 'lot', 'explain', 'most', 'side', 'because', 'thing', 'people', 'damage', 'happy', 'progress', 'hear', 'oh', 'short', 'happen', 'shrinkage', 'age', 'why', 'fine', 'ahead', 'think', 'many', 'again', 'whatever', 'quite', 'rim', 'move', 'ageing', 'go', 'wound', 'okay', 'computer', 'nobody', 'mind', 'memory', 'whatsoever', 'like', 'fair', 'pen', 'fact', 'short-term', 'one', 'pointer', 'chunk', 'anything', 'big', 'everything', 'tell', 'common', 'piece', 'look', 'yeah', 'really', 'middle', 'very', 'comment', 'foretell', 'criterion', 'fire', 'just', 'more', 'actually', 'miss', 'area', 'recent', 'no', 'well', 'little', 'is', 'how', 'fulfil', 'highlight', 'much', 'but', 'yes', 'word', 'from', 'time', 'it', 'describe', 'change', 'issue', 'tend', 'talk', 'twelve', 'or', 'reserve', 'over', 'future', 'sort', 'other', 'eight', 'bit', 'space', 'alzheimer', 'any', 'relevance', 'use', 'learn', 'once', 'extra', 'new', 'a', 'these', 'be', 'year', 'away', 'clearly', 'hall', 'question', 'way', 'around', 'suppose', 'that', 'and', 'brain', 'say', 'lucky', 'what', 'to', 'know', 'less', 'scan', 'not', 'become', 'fail', 'normal', 'enough', 'of', 'small', 'by', 'this', 'have', 'in', 'obviously', 'ten', 'those', 'vessel', 'good', 'up', 'will', 'combination', 'rather', 'should', 'if', 'so', 'plan', 'interesting', 'chat', 'let', 'now', 'imply', 'the', 'image', 'information', 'get', 'particular', 'test', 'show', 'about', 'strong', 'seventy', 'would', 'two', 'eighty', 'grey', 'at', 'last', 'always', 'blood', 'on', 'first', 'light', 'can', 'point', 'family', 'take', 'between', 'must', 'than', 'dr', 'honest', 'which', 'do', 'seem', 'an', 'all', 'black', '10', ' ', '  ', '   ', 'johnson', 'gosh', 'when', 'far', 'mean', 'with', 'absolutely', 'for', 'make', 'as', 'somewhere', 'screen', 'true', '20', 'correct', 'into', 'specifically', '90', 'dark', 'start', 'bottom', 'then', 'd', '100', 'out', 'line', 'where', 'pass', 'ct', 'i', 'round', 'open', 'mrs', 'clog']
vectors = []
for word in words:
    vectors.append(nlp(word).vector)
vectors = array(vectors)
st.sidebar.subheader('Embedding Visualization')
n_words = st.sidebar.slider("Number of words",1, len(words), 30)
words = array(words[:n_words])
vectors = vectors[:n_words, :]
reducer = UMAP(n_components=3)
scaled_data = StandardScaler().fit_transform(vectors)
embedding = reducer.fit_transform(scaled_data)
fig = scatter_3d(x = embedding[:, 0], y = embedding[:, 1], z = embedding[:, 2], text = words, hover_name=words)
st.plotly_chart(fig)
