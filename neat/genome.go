package neat

import (
	"fmt"

	"github.com/jmwri/neatgo/v2/network"
)

type Layers [][]network.Node

func (l Layers) Nodes() []network.Node {
	nodes := make([]network.Node, 0)
	for _, layer := range l {
		nodes = append(nodes, layer...)
	}
	return nodes
}

type Genome struct {
	Layers      Layers
	Connections []network.Connection
}

func (g Genome) NumLayers() int {
	return len(g.Layers)
}
func (g Genome) NumNodes() int {
	nodes := 0
	for _, layer := range g.Layers {
		nodes += len(layer)
	}
	return nodes
}
func (g Genome) NumConnections() int {
	return len(g.Connections)
}

// NumGenes is the total number of node and connection genes, which is the
// genome size used to normalise the compatibility distance.
func (g Genome) NumGenes() int {
	return g.NumNodes() + g.NumConnections()
}

// Compile builds a runnable network from the genome. The result is immutable
// and safe to activate from many goroutines at once, so compile once and reuse
// it rather than compiling per activation.
func (g Genome) Compile() (*network.Network, error) {
	return network.Compile(g.Layers.Nodes(), g.Connections)
}

func NewGenome(layers [][]network.Node, connections []network.Connection) Genome {
	return Genome{
		Layers:      layers,
		Connections: connections,
	}
}

// NewGenome builds a minimal genome: every layer described by cfg.Layers,
// fully connected between adjacent layers.
//
// Input and bias nodes are structural, not learnable. Inputs pass their value
// through untouched and bias nodes emit a constant 1, so both are created with
// a zero bias and are never mutated. Giving an input node a bias or a squashing
// activation would corrupt the very signal the network is meant to read.
func (b *Breeder) NewGenome() (Genome, error) {
	cfg, rng := b.cfg, b.rng
	genome := Genome{}
	if len(cfg.Layers) < 2 {
		return genome, fmt.Errorf("must have at least an input and output layer")
	}
	layers := make([][]network.Node, len(cfg.Layers))
	connections := make([]network.Connection, 0)
	for i, numNodes := range cfg.Layers {
		nodeType := network.Hidden
		if i == 0 {
			nodeType = network.Input
		}
		if i == len(cfg.Layers)-1 {
			nodeType = network.Output
		}

		for nodeNum := 0; nodeNum < numNodes; nodeNum++ {
			var bias float64
			var activationFn network.ActivationFunctionName
			switch nodeType {
			case network.Input:
				activationFn = cfg.InputActivationFn
			case network.Output:
				bias = cfg.RandBias(rng)
				activationFn = cfg.OutputActivationFn
			default:
				bias = cfg.RandBias(rng)
				activationFn = network.RandomActivationFunction(rng, cfg.HiddenActivationFns...)
			}
			node := network.NewNode(
				b.innovations.NodeID(),
				nodeType,
				bias,
				activationFn,
			)
			layers[i] = append(layers[i], node)
			if i > 0 {
				previousLayer := layers[i-1]
				for _, fromNode := range previousLayer {
					connection := network.NewConnection(
						b.innovations.ConnectionID(fromNode.ID, node.ID),
						fromNode.ID,
						node.ID,
						cfg.RandWeight(rng),
						true,
					)
					connections = append(connections, connection)
				}
			}
		}

		if i > 0 {
			continue
		}
		for nodeNum := 0; nodeNum < cfg.BiasNodes; nodeNum++ {
			node := network.NewNode(b.innovations.NodeID(), network.Bias, 0, network.NoActivation)
			layers[i] = append(layers[i], node)
		}
	}

	genome.Layers = layers
	genome.Connections = connections
	return genome, nil
}

func CopyGenome(genome Genome) Genome {
	cp := Genome{
		Layers:      make([][]network.Node, len(genome.Layers)),
		Connections: make([]network.Connection, len(genome.Connections)),
	}

	for i, layer := range genome.Layers {
		cp.Layers[i] = make([]network.Node, len(layer))
		copy(cp.Layers[i], layer)
	}
	copy(cp.Connections, genome.Connections)
	return cp
}

// RandomizeWeights returns a copy of genome with freshly sampled weights
// and biases but identical structure and identical gene IDs.
//
// This is how an initial population is seeded. Every member must share one
// genotype so that their genes line up under crossover and the compatibility
// distance sees them as one species; only the weights differ.
func (b *Breeder) RandomizeWeights(genome Genome) Genome {
	cfg, rng := b.cfg, b.rng
	genome = CopyGenome(genome)
	for i, layer := range genome.Layers {
		for j, node := range layer {
			if !mutableNode(node) {
				continue
			}
			genome.Layers[i][j].Bias = cfg.RandBias(rng)
		}
	}
	for i := range genome.Connections {
		genome.Connections[i].Weight = cfg.RandWeight(rng)
	}
	return genome
}

// mutableNode reports whether a node carries learnable parameters. Input and
// bias nodes do not: they are fixed signal sources.
func mutableNode(node network.Node) bool {
	return node.Type == network.Hidden || node.Type == network.Output
}

// MutateGenome returns a mutated copy of genome, leaving the original alone.
func (b *Breeder) MutateGenome(genome Genome) Genome {
	// Copy once. Each mutation below operates in place on this copy.
	return b.mutateGenome(CopyGenome(genome))
}

// mutateGenome mutates a genome the caller already owns exclusively, without
// copying it first. Crossover hands back a freshly built genome that nothing
// else references, so copying it again before mutating would duplicate every
// node and connection for nothing.
func (b *Breeder) mutateGenome(genome Genome) Genome {
	return b.mutateStructure(b.mutateParameters(genome))
}

// mutateParameters applies the mutations that only change existing genes.
//
// Nothing here allocates a historical marking, so these can run for many
// genomes at once. They deliberately come before any structural mutation: a
// node added this generation should start from the weights its split gave it,
// not be perturbed again on the way out.
func (b *Breeder) mutateParameters(genome Genome) Genome {
	genome = b.mutateNodeBiases(genome)
	genome = b.mutateNodeActivations(genome)
	genome = b.mutateConnectionWeights(genome)
	genome = b.mutateToggleEnabled(genome)
	return genome
}

// mutateStructure applies the mutations that add or remove genes.
//
// Adding a gene draws a historical marking from the shared innovation
// registry, and the marking a genome gets has to be the same on every run, so
// these are applied one genome at a time in a fixed order.
func (b *Breeder) mutateStructure(genome Genome) Genome {
	genome = b.mutateAddNode(genome)
	genome = b.mutateDeleteNode(genome)
	genome = b.mutateAddConnection(genome)
	genome = b.mutateDeleteConnection(genome)
	return genome
}

func getNodeFromLayers(layers [][]network.Node, nodeID int) (network.Node, bool) {
	for _, layer := range layers {
		for _, node := range layer {
			if node.ID == nodeID {
				return node, true
			}
		}
	}
	return network.Node{}, false
}

func getBiasNodes(layers [][]network.Node) []network.Node {
	nodes := make([]network.Node, 0)
	for _, layer := range layers {
		for _, node := range layer {
			if node.Type == network.Bias {
				nodes = append(nodes, node)
			}
		}
	}
	return nodes
}

func getNodeLayer(layers [][]network.Node, nodeID int) int {
	for i, layer := range layers {
		for _, node := range layer {
			if node.ID == nodeID {
				return i
			}
		}
	}
	return -1
}

// Crossover produces a child from two parents, best being the fitter of the two.
//
// Following the NEAT paper: genes present in both parents (matching genes,
// identified by their shared historical marking) take their value from either
// parent, while disjoint and excess genes are inherited from the fitter parent
// only. The child therefore has exactly the fitter parent's structure, which
// also keeps its input and output nodes in their original order - permuting
// them would silently rewire which input feeds which sensor.
func (b *Breeder) Crossover(best, worst Genome) Genome {
	cfg, rng := b.cfg, b.rng
	worstNodes := make(map[int]network.Node, worst.NumNodes())
	for _, layer := range worst.Layers {
		for _, node := range layer {
			worstNodes[node.ID] = node
		}
	}
	worstConnections := make(map[int]network.Connection, len(worst.Connections))
	for _, connection := range worst.Connections {
		worstConnections[connection.ID] = connection
	}

	childLayers := make(Layers, len(best.Layers))
	for i, layer := range best.Layers {
		childLayers[i] = make([]network.Node, len(layer))
		for j, bestNode := range layer {
			node := bestNode
			// Matching gene: take the parameters from either parent.
			if worstNode, ok := worstNodes[bestNode.ID]; ok && !cfg.Chance(rng, cfg.MateBestRate) {
				node.Bias = worstNode.Bias
				node.ActivationFn = worstNode.ActivationFn
			}
			childLayers[i][j] = node
		}
	}

	childConnections := make([]network.Connection, len(best.Connections))
	for i, bestConnection := range best.Connections {
		connection := bestConnection
		worstConnection, matching := worstConnections[bestConnection.ID]
		if matching && !cfg.Chance(rng, cfg.MateBestRate) {
			connection.Weight = worstConnection.Weight
		}
		if matching && (!bestConnection.Enabled || !worstConnection.Enabled) {
			// A gene disabled in either parent is usually, but not always,
			// disabled in the child. The occasional re-enable is what lets a
			// lineage recover a connection an add-node mutation switched off.
			connection.Enabled = !cfg.Chance(rng, cfg.MateDisabledRate)
		}
		childConnections[i] = connection
	}

	return Genome{
		Layers:      childLayers,
		Connections: childConnections,
	}
}
