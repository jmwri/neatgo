package neat

import (
	"github.com/jmwri/neatgo/network"
	"github.com/jmwri/neatgo/util"
)

func MutateAddConnection(cfg GenomeConfig, genome Genome) Genome {
	genome = CopyGenome(genome)
	seed := cfg.RandFloatProvider(0, 1)
	if seed > cfg.AddConnectionMutationRate {
		return genome
	}

	potentialConnections := getPotentialConnections(genome)
	if len(potentialConnections) == 0 {
		return genome
	}
	connectionToAdd := potentialConnections[util.IntBetween(0, len(potentialConnections))]
	connection := network.NewConnection(
		cfg.IDProvider.Next(),
		connectionToAdd.from,
		connectionToAdd.to,
		util.FloatBetween(cfg.MinWeight, cfg.MaxWeight),
		true,
	)

	genome.connections = append(genome.connections, connection)
	return genome
}

type potentialConnection struct {
	from, to int
}

func getPotentialConnections(genome Genome) []potentialConnection {
	existingConnections := make(map[int][]int)
	for _, connection := range genome.connections {
		existingConnections[connection.From] = append(existingConnections[connection.From], connection.To)
	}

	potentialConnections := make([]potentialConnection, 0)
	for i := 1; i < len(genome.layers); i++ {
		layer := genome.layers[i]
		previousLayer := genome.layers[i-1]
		for _, fromNode := range previousLayer {
			for _, toNode := range layer {
				if util.InSlice(existingConnections[fromNode.ID], toNode.ID) {
					continue
				}
				potentialConnections = append(potentialConnections, potentialConnection{from: fromNode.ID, to: toNode.ID})
			}
		}
	}

	return potentialConnections
}
