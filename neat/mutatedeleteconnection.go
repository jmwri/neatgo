package neat

import (
	"github.com/jmwri/neatgo/v2/internal/util"
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
	biasNodes := make(map[int]struct{})
	for _, node := range getBiasNodes(genome.Layers) {
		biasNodes[node.ID] = struct{}{}
	}
	deletableConnections := make([]int, 0, len(genome.Connections))
	for i, connection := range genome.Connections {
		if _, ok := biasNodes[connection.From]; ok {
			continue
		}
		if _, ok := biasNodes[connection.To]; ok {
			continue
		}
		deletableConnections = append(deletableConnections, i)
	}
	if len(deletableConnections) == 0 {
		return -1
	}
	return util.RandSliceElement(rng, deletableConnections)
}
