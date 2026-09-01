package neat_test

import (
	"fmt"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
)

func TestMutateNodeActivations_NoChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1, 1)
	cfg.ActivationMutationRate = 0
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	genome, err := breeder.NewGenome()
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := breeder.MutateNodeActivations(genome)
	assert.Equal(t, fmt.Sprint(genome), fmt.Sprint(actual))
}

func TestMutateNodeActivations_FullMutation(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1, 1)
	cfg.ActivationMutationRate = 1
	// A single choice, distinct from what the hidden node starts with, so the
	// mutation is observable rather than a coin flip.
	cfg.HiddenActivationFns = []network.ActivationFunctionName{network.Tanh}
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)

	layers := [][]network.Node{
		{{ID: 1, Type: network.Input, ActivationFn: network.NoActivation}},
		{{ID: 2, Type: network.Hidden, ActivationFn: network.Sigmoid}},
		{{ID: 3, Type: network.Output, ActivationFn: network.Sigmoid}},
	}
	genome := neat.NewGenome(layers, nil)

	actual := breeder.MutateNodeActivations(genome)
	assert.Equal(t, network.Tanh, actual.Layers[1][0].ActivationFn)
}

// Input, bias and output activations are fixed by the config. Mutating an input
// node's activation would squash the signal the network is supposed to read,
// and mutating the output's would change the range the caller reads results in.
func TestMutateNodeActivations_OnlyMutatesHiddenNodes(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.ActivationMutationRate = 1
	cfg.HiddenActivationFns = []network.ActivationFunctionName{network.Tanh}
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)

	layers := [][]network.Node{
		{
			{ID: 1, Type: network.Input, ActivationFn: network.NoActivation},
			{ID: 2, Type: network.Bias, ActivationFn: network.NoActivation},
		},
		{{ID: 3, Type: network.Output, ActivationFn: network.Sigmoid}},
	}
	genome := neat.NewGenome(layers, nil)

	actual := breeder.MutateNodeActivations(genome)
	assert.Equal(t, network.NoActivation, actual.Layers[0][0].ActivationFn, "input activation should not mutate")
	assert.Equal(t, network.NoActivation, actual.Layers[0][1].ActivationFn, "bias activation should not mutate")
	assert.Equal(t, network.Sigmoid, actual.Layers[1][0].ActivationFn, "output activation should not mutate")
}
