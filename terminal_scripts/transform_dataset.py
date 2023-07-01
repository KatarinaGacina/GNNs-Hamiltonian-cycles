from hamgnn.data.InMemoryDataset import ErdosRenyiInMemoryDataset
from hamgnn.constants import \
    GRAPH_DATA_DIRECTORY_SIZE_GENERALISATION,\
    GRAPH_DATA_DIRECTORY_SIZE_GENERALISATION2

#print(GRAPH_DATA_DIRECTORY_SIZE_GENERALISATION)
#print(GRAPH_DATA_DIRECTORY_SIZE_GENERALISATION2)

#adding edge attributes as node distances in two-dimensional layout created with ForceAtlas2
ErdosRenyiInMemoryDataset.transform(GRAPH_DATA_DIRECTORY_SIZE_GENERALISATION, GRAPH_DATA_DIRECTORY_SIZE_GENERALISATION2)

