package neat

import (
	"github.com/jmwri/neatgo/network"
	"github.com/jmwri/neatgo/util"
	"math/rand"
)

func MutateAddNode(cfg Config, genome Genome) Genome {
	genome = CopyGenome(genome)
	seed := cfg.RandFloatProvider(0, 1)
	if seed > cfg.AddNodeMutationRate {
		return genome
	}

	// If no Connections then we can't add a new node.
	// Add a new connection instead.
	if len(genome.Connections) == 0 {
		return MutateAddConnection(cfg, genome)
	}

	connectionIndex := getValidConnectionIndexForAddNodeMutation(genome)
	// If there are no Connections we can break, add a connection.
	if connectionIndex == -1 {
		return MutateAddConnection(cfg, genome)
	}

	connection := genome.Connections[connectionIndex]

	node := network.NewNode(
		cfg.IDProvider.Next(),
		network.Hidden,
		util.FloatBetween(cfg.MinBias, cfg.MaxBias),
		network.RandomActivationFunction(),
	)
	connectionFrom := network.NewConnection(
		cfg.IDProvider.Next(),
		connection.From,
		node.ID,
		util.FloatBetween(cfg.MinWeight, cfg.MaxWeight),
		true,
	)
	connectionTo := network.NewConnection(
		cfg.IDProvider.Next(),
		node.ID,
		connection.To,
		util.FloatBetween(cfg.MinWeight, cfg.MaxWeight),
		true,
	)

	// Figure out if we need to create a new layer
	fromLayer := getNodeLayer(genome.Layers, connectionFrom.From)
	toLayer := getNodeLayer(genome.Layers, connectionTo.To)
	// Calculate how many Layers there are between the connected nodes
	// From = 3
	// To = 4
	// layersBetween = 4-3-1 = 0
	// There are no Layers we can add a node to in between them, so need to create a new one!
	layersBetween := toLayer - fromLayer - 1
	// Always add to the layer closest to connectionFrom.From
	addToLayer := fromLayer + 1
	if layersBetween < 1 {
		// Shift all Layers from addToLayer up 1
		genome.Layers = append(genome.Layers[:addToLayer+1], genome.Layers[addToLayer:]...)
		genome.Layers[addToLayer] = []network.Node{}
	}

	// Disable old connection
	genome.Connections[connectionIndex].Enabled = false
	// Add new node + Connections to genome
	genome.Layers[addToLayer] = append(genome.Layers[addToLayer], node)
	genome.Connections = append(genome.Connections, connectionFrom)
	genome.Connections = append(genome.Connections, connectionTo)

	// If we're adding to the first layer after input, connect bias nodes to the new node.
	if addToLayer == 1 {
		for _, biasNode := range getBiasNodes(genome.Layers) {
			biasConnection := network.NewConnection(
				cfg.IDProvider.Next(),
				biasNode.ID,
				node.ID,
				util.FloatBetween(cfg.MinWeight, cfg.MaxWeight),
				true,
			)
			genome.Connections = append(genome.Connections, biasConnection)
		}
	}

	return genome
}

func getValidConnectionIndexForAddNodeMutation(genome Genome) int {
	// Build slice of Connections to process in order.
	// Shuffle the slice.
	connectionIndices := make([]int, len(genome.Connections))
	for i, _ := range genome.Connections {
		connectionIndices[i] = i
	}
	rand.Shuffle(len(connectionIndices), func(i, j int) {
		connectionIndices[i], connectionIndices[j] = connectionIndices[j], connectionIndices[i]
	})

	// Try each connection and return the first valid connection.
	for _, i := range connectionIndices {
		connection := genome.Connections[i]
		from := getNodeFromLayers(genome.Layers, connection.From)
		to := getNodeFromLayers(genome.Layers, connection.To)
		if from.Type == network.Bias || to.Type == network.Bias {
			// Don't break any bias Connections
			continue
		}
		return i
	}
	// No Connections are valid
	return -1
}
