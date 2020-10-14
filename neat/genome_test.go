package neat_test

import (
	"github.com/jmwri/neatgo/activation"
	"github.com/jmwri/neatgo/aggregation"
	"github.com/jmwri/neatgo/neat"
	"github.com/jmwri/neatgo/net"
	"github.com/jmwri/neatgo/util"
	"testing"
)

func TestGenome_Mutate(t *testing.T) {
	var n net.NeuralNetwork
	layerDefinitions := []net.LayerDefinition{
		net.NewLayerDefinition(2, 1, 1, 1, 1, activation.Nil, aggregation.Sum),
		net.NewLayerDefinition(3, 1, 1, 1, 1, activation.Nil, aggregation.Sum),
		net.NewLayerDefinition(2, 1, 1, 1, 1, activation.Nil, aggregation.Sum),
	}
	n, _ = net.NewFeedForwardFromDefinition(1, layerDefinitions)

	g1 := neat.NewGenome(n)
	outputA, _ := g1.Activate([]float64{1, 1})
	cfg := &neat.Config{
		WeightMutateRate: 1,
	}
	g1.Mutate(cfg)
	outputB, _ := g1.Activate([]float64{1, 1})
	if util.SliceOfFloatEqual(outputA, outputB) {
		t.Error("outputs are the same after mutation")
	}
}
