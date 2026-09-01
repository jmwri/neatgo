package neat_test

import (
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
)

func twoOutputGenome(outputBias float64, weight float64) neat.Genome {
	return neat.NewGenome(
		[][]network.Node{
			{
				{ID: 1, Type: network.Input, ActivationFn: network.NoActivation},
				{ID: 2, Type: network.Input, ActivationFn: network.NoActivation},
			},
			{
				{ID: 3, Type: network.Output, Bias: outputBias, ActivationFn: network.Sigmoid},
				{ID: 4, Type: network.Output, Bias: outputBias, ActivationFn: network.Sigmoid},
			},
		},
		[]network.Connection{
			{ID: 5, From: 1, To: 3, Weight: weight, Enabled: true},
			{ID: 6, From: 2, To: 4, Weight: weight, Enabled: true},
		},
	)
}

// The child must keep the fitter parent's node ordering. Nodes are read
// positionally - output[0] is the first output node in the last layer - so
// reordering them silently rewires which sensor feeds which input and which
// output the caller reads as which.
func TestCrossover_PreservesNodeOrder(t *testing.T) {
	cfg := neat.DefaultConfig(2, 2)
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	best := twoOutputGenome(1, .5)
	worst := twoOutputGenome(-1, -.5)

	for i := 0; i < 100; i++ {
		child := breeder.Crossover(best, worst)
		assert.Equal(t, nodeIDs(best), nodeIDs(child), "child node order must match the fitter parent")
		assert.Equal(t, connectionIDs(best), connectionIDs(child), "child connection order must match the fitter parent")
	}
}

// Matching genes may take their value from either parent, but the child must
// only ever contain genes the fitter parent has.
func TestCrossover_TakesValuesFromBothParents(t *testing.T) {
	cfg := neat.DefaultConfig(2, 2)
	cfg.MateBestRate = .5
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	best := twoOutputGenome(1, .5)
	worst := twoOutputGenome(-1, -.5)

	fromBest, fromWorst := 0, 0
	for i := 0; i < 200; i++ {
		child := breeder.Crossover(best, worst)
		for _, connection := range child.Connections {
			switch connection.Weight {
			case .5:
				fromBest++
			case -.5:
				fromWorst++
			default:
				t.Fatalf("child inherited a weight from neither parent: %v", connection.Weight)
			}
		}
	}
	assert.Greater(t, fromBest, 0, "some genes should come from the fitter parent")
	assert.Greater(t, fromWorst, 0, "some genes should come from the less fit parent")
}

// Genes the fitter parent does not have are excess or disjoint and are not
// inherited, so the child can never end up with a connection referencing a node
// it does not have.
func TestCrossover_DropsGenesMissingFromFitterParent(t *testing.T) {
	cfg := neat.DefaultConfig(2, 2)
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	best := twoOutputGenome(1, .5)
	worst := twoOutputGenome(-1, -.5)
	worst.Layers[1] = append(worst.Layers[1], network.Node{ID: 99, Type: network.Output, ActivationFn: network.Sigmoid})
	worst.Connections = append(worst.Connections, network.Connection{ID: 100, From: 1, To: 99, Weight: 2, Enabled: true})

	for i := 0; i < 100; i++ {
		child := breeder.Crossover(best, worst)
		assert.NotContains(t, nodeIDs(child), 99)
		assert.NotContains(t, connectionIDs(child), 100)

		known := make(map[int]bool)
		for _, id := range nodeIDs(child) {
			known[id] = true
		}
		for _, connection := range child.Connections {
			assert.True(t, known[connection.From], "connection references a node the child does not have")
			assert.True(t, known[connection.To], "connection references a node the child does not have")
		}
	}
}
