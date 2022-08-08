package neat

import (
	"fmt"
	"github.com/jmwri/neatgo/network"
	"github.com/jmwri/neatgo/util"
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
	layers      Layers
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

func GenerateGenome(cfg Config) (Genome, error) {
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

func MutateGenome(cfg Config, genome Genome) Genome {
	genome = MutateConnectionWeights(cfg, genome)
	genome = MutateAddConnection(cfg, genome)
	genome = MutateAddNode(cfg, genome)
	return genome
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

func Crossover(cfg Config, best, worst Genome) Genome {
	childLayers := make(Layers, len(best.layers))
	childConnections := make([]network.Connection, 0)

	// Count the number of innovations in each genome
	bestInnovationCount := make(map[int]int)
	worstInnovationCount := make(map[int]int)
	for _, bestLayer := range best.layers {
		for _, bestNode := range bestLayer {
			bestInnovationCount[bestNode.ID]++
		}
	}
	for _, bestConnection := range best.connections {
		bestInnovationCount[bestConnection.ID]++
	}
	for _, worstLayer := range worst.layers {
		for _, worstNode := range worstLayer {
			worstInnovationCount[worstNode.ID]++
		}
	}
	for _, worstConnection := range worst.connections {
		worstInnovationCount[worstConnection.ID]++
	}

	// Map to store which parent to take the gene from. 1 = best, 2 = worst.
	innovationParentChoice := make(map[int]int)
	// Set all genes to inherit from best by default
	for innovationID, bestCount := range bestInnovationCount {
		if bestCount < 1 {
			continue
		}
		innovationParentChoice[innovationID] = 1
	}
	for innovationID, worstCount := range worstInnovationCount {
		if worstCount < 1 {
			continue
		}
		if innovationParentChoice[innovationID] == 0 {
			// Doesn't exist in best, so don't add it
		} else {
			// Exists in best + worst
			// Work out if we should take best or worst gene
			if util.FloatBetween(0, 1) < cfg.MateBestRate {
				innovationParentChoice[innovationID] = 1
			} else {
				innovationParentChoice[innovationID] = 2
			}
		}
	}

	bestNodes := make(map[int]network.Node)
	bestNodesLayer := make(map[int]int)
	for layerNum, layer := range best.layers {
		for _, node := range layer {
			bestNodes[node.ID] = node
			bestNodesLayer[node.ID] = layerNum
		}
	}
	bestConnections := make(map[int]network.Connection)
	for _, connection := range best.connections {
		bestConnections[connection.ID] = connection
	}
	worstNodes := make(map[int]network.Node)
	worstNodesLayer := make(map[int]int)
	for layerNum, layer := range worst.layers {
		for _, node := range layer {
			worstNodes[node.ID] = node
			worstNodesLayer[node.ID] = layerNum
		}
	}
	worstConnections := make(map[int]network.Connection)
	for _, connection := range worst.connections {
		worstConnections[connection.ID] = connection
	}

	// Add each node that is chosen from best
	for _, bestLayer := range best.layers {
		for _, bestNode := range bestLayer {
			parentChoice := innovationParentChoice[bestNode.ID]
			layer := bestNodesLayer[bestNode.ID]
			if parentChoice == 1 {
				childLayers[layer] = append(childLayers[layer], bestNode)
			}
		}
	}

	// Add each node that is chosen from worst
	for _, worstLayer := range worst.layers {
		for _, worstNode := range worstLayer {
			parentChoice := innovationParentChoice[worstNode.ID]
			layer := worstNodesLayer[worstNode.ID]
			// If the node exists in best, take the layer of best instead to preserve and structural changes.
			if bestLayer, ok := bestNodesLayer[worstNode.ID]; ok {
				layer = bestLayer
			}
			if parentChoice == 2 {
				childLayers[layer] = append(childLayers[layer], worstNode)
			}
		}
	}

	// Add each connection that is chosen from best
	for _, bestConnection := range best.connections {
		parentChoice := innovationParentChoice[bestConnection.ID]
		if parentChoice == 1 {
			childConnections = append(childConnections, bestConnection)
		}
	}
	// Add each connection that is chosen from worst
	for _, worstConnection := range worst.connections {
		parentChoice := innovationParentChoice[worstConnection.ID]
		if parentChoice == 2 {
			childConnections = append(childConnections, worstConnection)
		}
	}

	return Genome{
		layers:      childLayers,
		connections: childConnections,
	}
}
