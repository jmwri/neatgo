package neat

import (
	"fmt"
	"github.com/jmwri/neatgo/network"
	"github.com/jmwri/neatgo/util"
)

type GenomeConfig struct {
	Layers                    []int
	BiasNodes                 int
	IDProvider                IDProvider
	RandFloatProvider         util.RandFloatProvider
	MinBias                   float64
	MaxBias                   float64
	MinWeight                 float64
	MaxWeight                 float64
	WeightMutationRate        float64
	WeightFullMutationRate    float64
	AddConnectionMutationRate float64
	AddNodeMutationRate       float64
}

func DefaultGenomeConfig(layers ...int) GenomeConfig {
	return GenomeConfig{
		Layers:                    layers,
		BiasNodes:                 1,
		IDProvider:                NewSequentialIDProvider(),
		RandFloatProvider:         util.FloatBetween,
		MinBias:                   -1,
		MaxBias:                   1,
		MinWeight:                 -1,
		MaxWeight:                 1,
		WeightMutationRate:        .8,
		WeightFullMutationRate:    .1,
		AddConnectionMutationRate: .05,
		AddNodeMutationRate:       .01,
	}
}

type Genome struct {
	layers      [][]network.Node
	connections []network.Connection
}

func (g Genome) NumLayers() int {
	return len(g.layers)
}
func (g Genome) NumNodes() int {
	nodes := 0
	for _, layer := range g.layers {
		nodes += len(layer)
	}
	return nodes
}
func (g Genome) NumConnections() int {
	return len(g.connections)
}

func NewGenome(layers [][]network.Node, connections []network.Connection) Genome {
	return Genome{
		layers:      layers,
		connections: connections,
	}
}

func GenerateGenome(cfg GenomeConfig) (Genome, error) {
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
			bias := cfg.RandFloatProvider(cfg.MinBias, cfg.MaxBias)
			activationFn := network.RandomActivationFunction()
			node := network.NewNode(
				cfg.IDProvider.Next(),
				nodeType,
				bias,
				activationFn,
			)
			layers[i] = append(layers[i], node)
			if i > 0 {
				previousLayer := layers[i-1]
				for _, fromNode := range previousLayer {
					weight := cfg.RandFloatProvider(cfg.MinWeight, cfg.MaxWeight)
					connection := network.NewConnection(
						cfg.IDProvider.Next(),
						fromNode.ID,
						node.ID,
						weight,
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
			node := network.NewNode(cfg.IDProvider.Next(), network.Bias, 0, network.NoActivationFn)
			layers[i] = append(layers[i], node)
		}
	}

	genome.layers = layers
	genome.connections = connections
	return genome, nil
}

func CopyGenome(genome Genome) Genome {
	cp := Genome{
		layers:      make([][]network.Node, len(genome.layers)),
		connections: make([]network.Connection, len(genome.connections)),
	}

	for i, layer := range genome.layers {
		cp.layers[i] = make([]network.Node, len(layer))
		for j, node := range layer {
			cp.layers[i][j] = node
		}
	}
	for i, connection := range genome.connections {
		cp.connections[i] = connection
	}
	return cp
}

func getNodeFromLayers(layers [][]network.Node, nodeID int) network.Node {
	for _, layer := range layers {
		for _, node := range layer {
			if node.ID == nodeID {
				return node
			}
		}
	}
	return network.Node{}
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
