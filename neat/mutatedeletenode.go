package neat

import (
	"github.com/jmwri/neatgo/v2/internal/util"
	"github.com/jmwri/neatgo/v2/network"
)

// MutateDeleteNode returns a copy of genome with a hidden node removed.
func (b *Breeder) MutateDeleteNode(genome Genome) Genome {
	return b.mutateDeleteNode(CopyGenome(genome))
}

func (b *Breeder) mutateDeleteNode(genome Genome) Genome {
	cfg, rng := b.cfg, b.rng
	if !cfg.Chance(rng, cfg.DeleteNodeMutationRate) {
		return genome
	}

	nodeToDelete := getLayerIndicesForNodeDeletion(rng, genome)
	if nodeToDelete.layer == -1 || nodeToDelete.nodeIndex == -1 {
		return genome
	}

	removeNodeID := genome.Layers[nodeToDelete.layer][nodeToDelete.nodeIndex].ID

	// Remove the node from the layer, preserving the order of its neighbours.
	genome.Layers[nodeToDelete.layer] = util.RemoveSliceIndexOrdered(genome.Layers[nodeToDelete.layer], nodeToDelete.nodeIndex)

	// Rebuild all layers, excluding any empty ones.
	newLayers := make(Layers, 0, len(genome.Layers))
	for _, layer := range genome.Layers {
		if len(layer) == 0 {
			continue
		}
		newLayers = append(newLayers, layer)
	}
	genome.Layers = newLayers

	// Drop every connection to or from the node. Leaving one behind would
	// create a dangling gene that references a node the network no longer has.
	keptConnections := make([]network.Connection, 0, len(genome.Connections))
	for _, connection := range genome.Connections {
		if connection.To == removeNodeID || connection.From == removeNodeID {
			continue
		}
		keptConnections = append(keptConnections, connection)
	}
	genome.Connections = keptConnections

	return genome
}

type nodeLayerIndices struct {
	layer     int
	nodeIndex int
}

func getLayerIndicesForNodeDeletion(rng *Rand, genome Genome) nodeLayerIndices {
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
		return nodeLayerIndices{layer: -1, nodeIndex: -1}
	}

	return util.RandSliceElement(rng, nodesLayerIndices)
}
