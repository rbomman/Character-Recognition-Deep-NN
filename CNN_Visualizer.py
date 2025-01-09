import matplotlib.pyplot as plt
import networkx as nx

def visualize_cnn_structure():
    # Define the CNN structure based on the code provided
    # Shows the dimensions of the feature map
    layers = [
        ("Input", (28, 28, 1)),
        ("Conv2D (32 filters, 3x3)", (26, 26, 32)),
        ("MaxPool2D (2x2)", (13, 13, 32)),
        ("Conv2D (64 filters, 3x3)", (11, 11, 64)),
        ("MaxPool2D (2x2)", (5, 5, 64)),
        ("Flatten", (1600,)),
        ("Dense (128 units)", (128,)),
        ("Dense (10 units)", (10,))
    ]

    # Create a directed graph
    G = nx.DiGraph()
    previous_node = None
    pos = {}
    y = 0  # Vertical positioning for layers
    for layer_name, shape in layers:
        node_label = f"{layer_name}\n{shape}"
        G.add_node(node_label)
        pos[node_label] = (0, y)
        if previous_node:
            G.add_edge(previous_node, node_label)
        previous_node = node_label
        y -= 1

    # Draw the graph with labels
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, edge_color="gray")
    plt.title("CNN Architecture")
    # Save the figure as a PNG
    plt.savefig("cnn_architecture.png", dpi=300)  # <-- Added line to save the plot
    plt.show()

# Generate the visualization
visualize_cnn_structure()

