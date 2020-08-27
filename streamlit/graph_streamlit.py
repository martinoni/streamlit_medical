import srsly
import codecs
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as stc

def st_graph(html):
    graph_file = codecs.open(html, 'r')
    page = graph_file.read()
    stc.html(page, width=900, height=1000)

graph_file = srsly.read_json('graph.json')

st.title('Medical Graph')
st.sidebar.subheader('Similarity Threshold')
threshold = st.sidebar.slider("",0.0, 1.0)

graph = Network(width='100%',bgcolor="#F6F8FB", font_color="#444444", directed=True, heading='', notebook=True)
for link in graph_file['links']:
    similarity = float(link['weight'])
    if similarity >= threshold and similarity != 1.0:
        graph.add_node(link['source'], size=14, color = '#2169AD')
        graph.add_node(link['target'], size=10, color = '#78CABC')
        graph.add_edge(link['source'], link['target'], weight = round(similarity * 100))

#graph.show_buttons(filter_=['physics'])
graph.force_atlas_2based()
graph.show('graph.html')
st_graph('graph.html')

st.sidebar