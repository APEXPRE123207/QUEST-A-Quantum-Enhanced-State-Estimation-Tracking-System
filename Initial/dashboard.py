import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import networkx as nx
import pickle
import os

# Import our project's classes
from Quantum_Core.qneat import Population

# --- Graph Generation Logic ---
def generate_graph_figure(population: Population):
    """Creates a Plotly figure of the speciation graph."""
    G = nx.Graph()
    fig = go.Figure()
    
    if not population or not population.species:
        return fig # Return an empty figure if no data

    # Add nodes for each species and each genome
    species_nodes = []
    genome_nodes = []
    for i, species in enumerate(population.species):
        species_node_name = f"Species {i}"
        G.add_node(species_node_name, type='species')
        species_nodes.append(species_node_name)
        for genome in species.members:
            genome_node_name = f"Genome {id(genome)}"
            G.add_node(genome_node_name, type='genome')
            genome_nodes.append(genome_node_name)
            G.add_edge(species_node_name, genome_node_name)

    # Use a bipartite layout to separate species from genomes
    pos = nx.bipartite_layout(G, species_nodes)

    # --- Create the Plotly visualization ---
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#888')))

    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
    fig.add_trace(go.Scatter(x=node_x, y=node_y, text=node_text, mode='markers+text',
                           marker=dict(size=10, color='lightblue')))
    
    fig.update_layout(title_text=f"QNEAT Speciation - Generation {population.generation}",
                      showlegend=False, xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    return fig

# --- Dash App ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("QNEAT Real-Time Speciation"),
    dcc.Graph(id='speciation-graph'),
    dcc.Interval(id='interval-component', interval=2*1000, n_intervals=0) # Update every 2 seconds
])

@app.callback(Output('speciation-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph(n):
    """Callback to reload data and update the graph."""
    state_file = "population_state.pkl"
    if os.path.exists(state_file):
        try:
            with open(state_file, "rb") as f:
                population = pickle.load(f)
            return generate_graph_figure(population)
        except Exception as e:
            print(f"Error loading state file: {e}")
    return go.Figure() # Return empty figure on error

if __name__ == '__main__':
    app.run(debug=True)