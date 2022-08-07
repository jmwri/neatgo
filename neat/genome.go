package neat

import (
	"fmt"
	"github.com/jmwri/neatgo/network"
)

type GenomeConfig struct {
	Layers                    []int
	BiasNodes                 int
	IDProvider                IDProvider
	RandFloatProvider         RandFloatProvider
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
		RandFloatProvider:         FloatBetween,
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
	nodes       []network.Node
	connections []network.Connection
}

func GenerateGenome(cfg GenomeConfig) (Genome, error) {
	genome := Genome{}
	if len(cfg.Layers) < 2 {
		return genome, fmt.Errorf("must have at least an input and output layer")
	}
	nodes := make([][]network.Node, len(cfg.Layers))
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
			activationFn := func(x float64) float64 { return x }
			node := network.NewNode(
				cfg.IDProvider.Next(),
				nodeType,
				bias,
				activationFn,
			)
			nodes[i] = append(nodes[i], node)
			if i > 0 {
				nodesInPreviousLayer := nodes[i-1]
				for _, fromNode := range nodesInPreviousLayer {
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
	}

	allNodes := make([]network.Node, 0)
	for _, layerNodes := range nodes {
		for _, node := range layerNodes {
			allNodes = append(allNodes, node)
		}
	}

	genome.nodes = allNodes
	genome.connections = connections
	return genome, nil
}

func CopyGenome(genome Genome) Genome {
	cp := Genome{
		nodes:       make([]network.Node, len(genome.nodes)),
		connections: make([]network.Connection, len(genome.connections)),
	}

	for i, node := range genome.nodes {
		cp.nodes[i] = node
	}
	for i, connection := range genome.connections {
		cp.connections[i] = connection
	}
	return cp
}
