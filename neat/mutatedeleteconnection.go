package neat

import (
	"github.com/jmwri/neatgo/v2/internal/util"
	"github.com/jmwri/neatgo/v2/network"
)

// MutateDeleteConnection returns a copy of genome with a connection removed.
func (b *Breeder) MutateDeleteConnection(genome Genome) Genome {
	return b.mutateDeleteConnection(CopyGenome(genome))
}

func (b *Breeder) mutateDeleteConnection(genome Genome) Genome {
	cfg, rng := b.cfg, b.rng
	if !cfg.Chance(rng, cfg.DeleteConnectionMutationRate) {
		return genome
	}

	connectionToDelete := getConnectionIndexForDeletion(rng, genome)
	if connectionToDelete == -1 {
		return genome
	}

	genome.Connections = util.RemoveSliceIndexOrdered(genome.Connections, connectionToDelete)

	return genome
}

func getConnectionIndexForDeletion(rng *Rand, genome Genome) int {
	deletableConnections := make([]int, 0)
	for i, connection := range genome.Connections {
		fromNode, fromOK := getNodeFromLayers(genome.Layers, connection.From)
		toNode, toOK := getNodeFromLayers(genome.Layers, connection.To)
		if fromOK && fromNode.Type == network.Bias {
			continue
		}
		if toOK && toNode.Type == network.Bias {
			continue
		}
		deletableConnections = append(deletableConnections, i)
	}
	if len(deletableConnections) == 0 {
		return -1
	}
	return util.RandSliceElement(rng, deletableConnections)
}
