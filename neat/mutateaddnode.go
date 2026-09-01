package neat

import (
	"github.com/jmwri/neatgo/v2/network"
)

// MutateAddNode returns a copy of genome with a connection split by a new node.
func (b *Breeder) MutateAddNode(genome Genome) Genome {
	return b.mutateAddNode(CopyGenome(genome))
}

// mutateAddNode splits an existing connection in two with a new hidden node
// between the halves, disabling the original connection.
//
// The split is deliberately close to neutral: the incoming half gets a weight
// of 1 and the outgoing half inherits the old weight, so the new node starts
// out reproducing roughly what the connection it replaced did. A split that
// randomises both weights is a large, usually fatal, perturbation - and NEAT
// depends on new structure surviving long enough to be optimised.
func (b *Breeder) mutateAddNode(genome Genome) Genome {
	cfg, rng := b.cfg, b.rng
	if !cfg.Chance(rng, cfg.AddNodeMutationRate) {
		return genome
	}

	connectionIndex := b.validConnectionForAddNode(rng, genome)
	if connectionIndex == -1 {
		return genome
	}

	connection := genome.Connections[connectionIndex]
	fromLayer := getNodeLayer(genome.Layers, connection.From)
	toLayer := getNodeLayer(genome.Layers, connection.To)
	if fromLayer == -1 || toLayer == -1 || toLayer <= fromLayer {
		// Dangling or non feed-forward connection; nothing safe to split.
		return genome
	}

	// The same split, discovered by any genome, yields the same three gene IDs.
	split := b.innovations.SplitConnection(connection.ID, connection.From, connection.To)

	node := network.NewNode(
		split.NodeID,
		network.Hidden,
		0,
		network.RandomActivationFunction(rng, cfg.HiddenActivationFns...),
	)
	connectionFrom := network.NewConnection(split.InConnection, connection.From, node.ID, 1, true)
	connectionTo := network.NewConnection(split.OutConnection, node.ID, connection.To, connection.Weight, true)

	// Always add to the layer closest to connection.From, inserting a new layer
	// if the two nodes are already adjacent.
	addToLayer := fromLayer + 1
	if toLayer-fromLayer-1 < 1 {
		genome.Layers = append(genome.Layers, nil)
		copy(genome.Layers[addToLayer+1:], genome.Layers[addToLayer:])
		genome.Layers[addToLayer] = []network.Node{}
	}

	genome.Connections[connectionIndex].Enabled = false
	genome.Layers[addToLayer] = append(genome.Layers[addToLayer], node)
	genome.Connections = append(genome.Connections, connectionFrom, connectionTo)

	return genome
}

func (b *Breeder) validConnectionForAddNode(rng *Rand, genome Genome) int {
	presentNodes := make(map[int]struct{}, genome.NumNodes())
	for _, layer := range genome.Layers {
		for _, node := range layer {
			presentNodes[node.ID] = struct{}{}
		}
	}

	// Build slice of Connections to process in order.
	// Shuffle the slice.
	connectionIndices := make([]int, len(genome.Connections))
	for i := range genome.Connections {
		connectionIndices[i] = i
	}
	rng.Shuffle(len(connectionIndices), func(i, j int) {
		connectionIndices[i], connectionIndices[j] = connectionIndices[j], connectionIndices[i]
	})

	// Try each connection and return the first valid connection.
	for _, i := range connectionIndices {
		connection := genome.Connections[i]
		if !connection.Enabled {
			// Splitting an already disabled connection would add a node that
			// contributes nothing.
			continue
		}
		from, ok := getNodeFromLayers(genome.Layers, connection.From)
		if !ok {
			continue
		}
		to, ok := getNodeFromLayers(genome.Layers, connection.To)
		if !ok {
			continue
		}
		if from.Type == network.Bias || to.Type == network.Bias {
			// Don't break any bias Connections
			continue
		}
		// Historical markings are stable, so splitting a given connection
		// always yields the same node ID. If this genome already holds that
		// node - because the connection was re-enabled after an earlier split -
		// splitting again would add a second copy of the same gene.
		if split, ok := b.innovations.LookupSplit(connection.ID); ok {
			if _, exists := presentNodes[split.NodeID]; exists {
				continue
			}
		}
		return i
	}
	// No Connections are valid
	return -1
}
