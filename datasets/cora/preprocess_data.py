def extracting_info(data):
    graph = data[0]

    features = graph.ndata['feat']
    labels   = graph.ndata['label']

    train_mask = graph.ndata['train_mask']
    val_mask   = graph.ndata['val_mask']
    test_mask  = graph.ndata['test_mask']

    masks = [train_mask, val_mask, test_mask]

    return graph, features, labels, masks

def to_adjacency(matrix):
    adjacency_matrix = None
    return adjacency_matrix
