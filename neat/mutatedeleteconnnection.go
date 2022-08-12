package neat

import (
	"github.com/jmwri/neatgo/network"
	"github.com/jmwri/neatgo/util"
)

func MutateDeleteConnection(cfg Config, genome Genome) Genome {
	genome = CopyGenome(genome)
	seed := cfg.RandFloatProvider(0, 1)
	if seed > cfg.DeleteConnectionMutationRate {
		return genome
	}

	connectionToDelete := getConnectionIndicesForDeletion(genome)
	if connectionToDelete == -1 {
		return genome
	}

	genome.Connections = util.RemoveSliceIndex(genome.Connections, connectionToDelete)

	return genome
}

func getConnectionIndicesForDeletion(genome Genome) int {
	deletableConnections := make([]int, 0)
	for i, connection := range genome.Connections {
		fromNode := getNodeFromLayers(genome.Layers, connection.From)
		toNode := getNodeFromLayers(genome.Layers, connection.From)
		if fromNode.Type == network.Bias || toNode.Type == network.Bias {
			continue
		}
		deletableConnections = append(deletableConnections, i)
	}
	if len(deletableConnections) == 0 {
		return -1
	}
	return util.RandSliceElement(deletableConnections)
}
