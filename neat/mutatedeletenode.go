package neat

import (
	"github.com/jmwri/neatgo/network"
	"github.com/jmwri/neatgo/util"
	"sort"
)

func MutateDeleteNode(cfg Config, genome Genome) Genome {
	genome = CopyGenome(genome)
	seed := cfg.RandFloatProvider(0, 1)
	if seed > cfg.DeleteNodeMutationRate {
		return genome
	}

	nodeToDelete := getLayerIndicesForNodeDeletion(genome)
	if nodeToDelete.layer == -1 || nodeToDelete.nodeIndex == -1 {
		return genome
	}

	removeNodeID := genome.Layers[nodeToDelete.layer][nodeToDelete.nodeIndex].ID

	// Remove the node from the layer.
	genome.Layers[nodeToDelete.layer] = util.RemoveSliceIndex(genome.Layers[nodeToDelete.layer], nodeToDelete.nodeIndex)

	// Rebuild all layers, excluding any empty ones.
	newLayers := make(Layers, 0)
	for _, layer := range genome.Layers {
		if len(layer) == 0 {
			continue
		}
		newLayers = append(newLayers, layer)
	}
	genome.Layers = newLayers

	// Gather all connections to/from the node
	removeConnectionIndices := make([]int, 0)
	for i, connection := range genome.Connections {
		if connection.To == removeNodeID || connection.From == removeNodeID {
			removeConnectionIndices = append(removeConnectionIndices, i)
		}
	}

	// Sort from highest to lowest, so when removing we don't modify any of the other indices
	sort.Slice(removeConnectionIndices, func(i, j int) bool {
		return removeConnectionIndices[i] > removeConnectionIndices[j]
	})

	// Remove each connection
	for _, connectionIndex := range removeConnectionIndices {
		genome.Connections = util.RemoveSliceIndex(genome.Connections, connectionIndex)
	}

	return genome
}

type nodeLayerIndices struct {
	layer     int
	nodeIndex int
}

func getLayerIndicesForNodeDeletion(genome Genome) nodeLayerIndices {
	// Build slice of NodeIDs to process in order.
	// Shuffle the slice.
	nodesLayerIndices := make([]nodeLayerIndices, 0)
	for i, layer := range genome.Layers {
		for j, node := range layer {
			if node.Type != network.Hidden {
				continue
			}
			nodesLayerIndices = append(nodesLayerIndices, nodeLayerIndices{
				layer:     i,
				nodeIndex: j,
			})
		}
	}

	// No nodes we can remove, so no mutation.
	if len(nodesLayerIndices) == 0 {
		return nodeLayerIndices{layer: -1}
	}

	return util.RandSliceElement(nodesLayerIndices)
}
