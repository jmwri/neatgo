package neat

import (
	"errors"
	"github.com/jmwri/neatgo/net"
)

var ErrCrossoverNotSameParent = errors.New("cant crossover gene if IDs do not match")

func GenesFromNetwork(n net.NeuralNetwork) ([][]*NodeGene, [][]*ConnectionGene) {
	layers := n.Layers()
	nodeGenes := make([][]*NodeGene, len(layers))
	for i := 0; i < len(layers); i++ {
		nodeGenes[i] = make([]*NodeGene, len(layers[i]))
		for ni, node := range layers[i] {
			nodeGenes[i][ni] = NewNodeGene(node)
		}
	}

	connections := n.Connections()
	connectionGenes := make([][]*ConnectionGene, len(connections))
	for i := 0; i < len(connections); i++ {
		connectionGenes[i] = make([]*ConnectionGene, len(connections[i]))
		for ci, connection := range connections[i] {
			connectionGenes[i][ci] = NewConnectionGene(connection)
		}
	}

	return nodeGenes, connectionGenes
}
