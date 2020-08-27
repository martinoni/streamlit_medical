import srsly
import codecs
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as stc

def st_graph(html):
    graph_file = codecs.open(html, 'r')
    page = graph_file.read()
    stc.html(page, width=700, height=700)

graph_file = srsly.read_json('graph.json')

st.title('Medical Graph')

graph = Network(height="750px", width="100%", bgcolor="#E5ECF6", font_color="#444444", directed=True, notebook=True)
for link in graph_file['links']:
    similarity = float(link['weight'])
    if similarity >= 0.2 and similarity != 1.0:
        graph.add_node(link['source'], size=14, color = '#2169AD')
        graph.add_node(link['target'], size=10, color = '#78CABC')
        graph.add_edge(link['source'], link['target'], weight = round(similarity * 100))

graph.show_buttons(filter_=['physics'])
graph.show('graph.html')
st_graph('graph.html')