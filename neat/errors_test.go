package neat_test

import (
	"context"
	"errors"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestErrors_AreDiscriminable(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 10

	t.Run("invalid config", func(t *testing.T) {
		bad := cfg
		bad.PopulationSize = 0
		_, err := neat.GeneratePopulation(bad)
		assert.ErrorIs(t, err, neat.ErrInvalidConfig)
	})

	t.Run("nil evaluator", func(t *testing.T) {
		pop, err := neat.GeneratePopulation(cfg)
		require.NoError(t, err)
		assert.ErrorIs(t, neat.Evaluate(context.Background(), pop, nil), neat.ErrNoEvaluator)
	})

	t.Run("no stop condition", func(t *testing.T) {
		pop, err := neat.GeneratePopulation(cfg)
		require.NoError(t, err)
		_, err = neat.Run(context.Background(), pop, constantFitness(1), neat.RunOptions{})
		assert.ErrorIs(t, err, neat.ErrNoStopCondition)
	})

	// An evaluator's own error must still be reachable, so a caller can react
	// to it rather than only knowing that "something failed".
	t.Run("evaluator error is wrapped, not replaced", func(t *testing.T) {
		pop, err := neat.GeneratePopulation(cfg)
		require.NoError(t, err)
		boom := errors.New("simulator crashed")
		_, err = neat.RunGeneration(context.Background(), pop, func(context.Context, *network.Network) (float64, error) {
			return 0, boom
		})
		assert.ErrorIs(t, err, neat.ErrEvaluate)
		assert.ErrorIs(t, err, boom)
	})
}

func TestNetworkErrors_AreDiscriminable(t *testing.T) {
	input := network.Node{ID: 1, Type: network.Input, ActivationFn: network.NoActivation}
	output := network.Node{ID: 2, Type: network.Output, ActivationFn: network.NoActivation}

	t.Run("cycle", func(t *testing.T) {
		nodes := []network.Node{input, {ID: 3, Type: network.Hidden, ActivationFn: network.NoActivation}, {ID: 4, Type: network.Hidden, ActivationFn: network.NoActivation}, output}
		connections := []network.Connection{
			{ID: 5, From: 3, To: 4, Weight: 1, Enabled: true},
			{ID: 6, From: 4, To: 3, Weight: 1, Enabled: true},
		}
		_, err := network.Compile(nodes, connections)
		assert.ErrorIs(t, err, network.ErrCycle)
	})

	t.Run("unknown activation", func(t *testing.T) {
		_, err := network.Compile([]network.Node{input, {ID: 2, Type: network.Output, ActivationFn: "nope"}}, nil)
		assert.ErrorIs(t, err, network.ErrUnknownActivation)
	})

	t.Run("duplicate node", func(t *testing.T) {
		_, err := network.Compile([]network.Node{input, input}, nil)
		assert.ErrorIs(t, err, network.ErrDuplicateNode)
	})

	t.Run("wrong input size", func(t *testing.T) {
		net, err := network.Compile([]network.Node{input, output}, nil)
		require.NoError(t, err)
		_, err = net.Activate([]float64{1, 2, 3})
		assert.ErrorIs(t, err, network.ErrInputSize)
	})

	t.Run("wrong output buffer", func(t *testing.T) {
		net, err := network.Compile([]network.Node{input, output}, nil)
		require.NoError(t, err)
		err = net.ActivateInto([]float64{1}, make([]float64, 5))
		assert.ErrorIs(t, err, network.ErrOutputSize)
	})

	// A genome with a cycle must surface as a compile failure through the
	// evolutionary loop too, not as a hang or a silent bad score.
	t.Run("surfaces through Evaluate", func(t *testing.T) {
		cfg := neat.DefaultConfig(2, 1)
		cfg.PopulationSize = 4
		pop, err := neat.GeneratePopulation(cfg)
		require.NoError(t, err)
		pop.Genomes[0].Connections = append(pop.Genomes[0].Connections,
			network.Connection{ID: 9999, From: pop.Genomes[0].Layers[1][0].ID, To: pop.Genomes[0].Layers[0][0].ID, Weight: 1, Enabled: true})

		err = neat.Evaluate(context.Background(), pop, constantFitness(1))
		assert.ErrorIs(t, err, neat.ErrCompile)
		assert.ErrorIs(t, err, network.ErrCycle)
	})
}
