package neat_test

import (
	"github.com/jmwri/neatgo/neat"
	"github.com/jmwri/neatgo/net"
	"github.com/jmwri/neatgo/util"
	"testing"
)

func TestGenome_Mutate(t *testing.T) {
	var n net.NeuralNetwork
	layerDefinitions := []net.LayerDefinition{
		net.NewLayerDefinition(2, 1, 1, nil, nil),
		net.NewLayerDefinition(3, 1, 1, nil, nil),
		net.NewLayerDefinition(2, 1, 1, nil, nil),
	}
	n, _ = net.NewFeedForwardFromDefinition(1, layerDefinitions)

	g1 := neat.NewGenome(n)
	outputA, _ := g1.Activate([]float64{1, 1})
	g1.Mutate()
	outputB, _ := g1.Activate([]float64{1, 1})
	if util.SliceOfFloatEqual(outputA, outputB) {
		t.Error("outputs are the same after mutation")
	}
}
