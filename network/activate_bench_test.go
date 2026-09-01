package network_test

import (
	"fmt"
	"testing"

	"github.com/jmwri/neatgo/v2/network"
)

// buildNetwork returns a fully connected feed-forward network with the given
// layer sizes.
func buildNetwork(layerSizes ...int) ([]network.Node, []network.Connection, []float64) {
	nodes := make([]network.Node, 0)
	connections := make([]network.Connection, 0)
	layers := make([][]network.Node, len(layerSizes))
	id := 0

	for i, size := range layerSizes {
		nodeType := network.Hidden
		if i == 0 {
			nodeType = network.Input
		} else if i == len(layerSizes)-1 {
			nodeType = network.Output
		}
		for j := 0; j < size; j++ {
			id++
			node := network.NewNode(id, nodeType, .1, network.Sigmoid)
			layers[i] = append(layers[i], node)
			nodes = append(nodes, node)
			if i == 0 {
				continue
			}
			for _, from := range layers[i-1] {
				id++
				connections = append(connections, network.NewConnection(id, from.ID, node.ID, .5, true))
			}
		}
	}

	input := make([]float64, layerSizes[0])
	for i := range input {
		input[i] = float64(i%3) - 1
	}
	return nodes, connections, input
}

func BenchmarkActivate(b *testing.B) {
	sizes := [][]int{
		{2, 1},
		{2, 3, 1},
		{8, 16, 16, 4},
		{32, 64, 64, 8},
	}
	for _, layerSizes := range sizes {
		nodes, connections, input := buildNetwork(layerSizes...)
		b.Run(fmt.Sprintf("%v/%dnodes/%dconns", layerSizes, len(nodes), len(connections)), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := network.Activate(nodes, connections, input); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
