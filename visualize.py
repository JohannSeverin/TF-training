import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3D

def plot_graph(graph):
    """
    Function to plot graph of the type made by spektral.
    The position of node are plotted, where the size is equal to the
    charges detected by the dome.

    Edges are made according to the adjacency matrix


    make colors for time???? maybe alpha

    """

    # Setup the figure
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = "3d")

    # Find important features from Graph
    pos  = graph.x[]
    size = graph.x

    a    = graph.a 
    send = np.arang
    rec  = a.indicies

    # Plot nodes
    ax.scatter(positions,, s = size)


    # Plot edges
    for i, j in zip(send, rec):
        plot()    


    return fig, ax
