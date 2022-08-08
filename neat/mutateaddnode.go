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

	// If no connections then we can't add a new node.
	// Add a new connection instead.
	if len(genome.connections) == 0 {
		return MutateAddConnection(cfg, genome)
	}

	connectionIndex := getValidConnectionIndexForAddNodeMutation(genome)
	// If there are no connections we can break, add a connection.
	if connectionIndex == -1 {
		return MutateAddConnection(cfg, genome)
	}

	connection := genome.connections[connectionIndex]

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
	fromLayer := getNodeLayer(genome.layers, connectionFrom.From)
	toLayer := getNodeLayer(genome.layers, connectionTo.To)
	// Calculate how many layers there are between the connected nodes
	// From = 3
	// To = 4
	// layersBetween = 4-3-1 = 0
	// There are no layers we can add a node to in between them, so need to create a new one!
	layersBetween := toLayer - fromLayer - 1
	// Always add to the layer closest to connectionFrom.From
	addToLayer := fromLayer + 1
	if layersBetween < 1 {
		// Shift all layers from addToLayer up 1
		genome.layers = append(genome.layers[:addToLayer+1], genome.layers[addToLayer:]...)
		genome.layers[addToLayer] = []network.Node{}
	}

	// Disable old connection
	genome.connections[connectionIndex].Enabled = false
	// Add new node + connections to genome
	genome.layers[addToLayer] = append(genome.layers[addToLayer], node)
	genome.connections = append(genome.connections, connectionFrom)
	genome.connections = append(genome.connections, connectionTo)

	// If we're adding to the first layer after input, connect bias nodes to the new node.
	if addToLayer == 1 {
		for _, biasNode := range getBiasNodes(genome.layers) {
			biasConnection := network.NewConnection(
				cfg.IDProvider.Next(),
				biasNode.ID,
				node.ID,
				util.FloatBetween(cfg.MinWeight, cfg.MaxWeight),
				true,
			)
			genome.connections = append(genome.connections, biasConnection)
		}
	}

	return genome
}

func getValidConnectionIndexForAddNodeMutation(genome Genome) int {
	// Build slice of connections to process in order.
	// Shuffle the slice.
	connectionIndices := make([]int, len(genome.connections))
	for i, _ := range genome.connections {
		connectionIndices[i] = i
	}
	rand.Shuffle(len(connectionIndices), func(i, j int) {
		connectionIndices[i], connectionIndices[j] = connectionIndices[j], connectionIndices[i]
	})

	// Try each connection and return the first valid connection.
	for _, i := range connectionIndices {
		connection := genome.connections[i]
		from := getNodeFromLayers(genome.layers, connection.From)
		to := getNodeFromLayers(genome.layers, connection.To)
		if from.Type == network.Bias || to.Type == network.Bias {
			// Don't break any bias connections
			continue
		}
		return i
	}
	// No connections are valid
	return -1
}
