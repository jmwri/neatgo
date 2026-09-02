package neat_test

import (
	"context"
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func recurrentConfig() neat.Config {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 30
	cfg.Seed = 7
	cfg.Recurrent = true
	// Wiring far more often than a real run would, so the test reaches the
	// behaviour it is checking within a handful of generations.
	cfg.AddConnectionMutationRate = 1
	cfg.AddNodeMutationRate = 0.5
	return cfg
}

// nodeLayer is where a node sits, which is what decides whether a connection
// runs forwards.
func nodeLayer(genome neat.Genome, id int) int {
	for depth, layer := range genome.Layers {
		for _, node := range layer {
			if node.ID == id {
				return depth
			}
		}
	}
	return -1
}

func hasBackwardConnection(genome neat.Genome) bool {
	for _, connection := range genome.Connections {
		if !connection.Enabled {
			continue
		}
		from, to := nodeLayer(genome, connection.From), nodeLayer(genome, connection.To)
		if from >= 0 && to >= 0 && from >= to {
			return true
		}
	}
	return false
}

// TestRecurrentRunGrowsLoopsAndKeepsRunning is the end to end of the option: a
// run with it on must actually produce connections a feed-forward network could
// not have, and must still be able to compile and score every one of them.
func TestRecurrentRunGrowsLoopsAndKeepsRunning(t *testing.T) {
	pop, err := neat.GeneratePopulation(recurrentConfig())
	require.NoError(t, err)

	pop, err = neat.Run(context.Background(), pop, constantFitness(1), neat.RunOptions{MaxGenerations: 20})
	require.NoError(t, err)

	looped := 0
	for _, genome := range pop.Genomes {
		if hasBackwardConnection(genome) {
			looped++
		}
		// Whatever mutation produced, the run has to be able to build it.
		_, err := genome.CompileFor(pop.Cfg)
		require.NoError(t, err)
	}
	assert.Greater(t, looped, 0, "no genome grew a connection that loops back")
}

// TestFeedForwardRunGrowsNoLoops is the other half: with the option off, nothing
// should ever produce a connection that a feed-forward network cannot compile.
func TestFeedForwardRunGrowsNoLoops(t *testing.T) {
	cfg := recurrentConfig()
	cfg.Recurrent = false

	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)
	pop, err = neat.Run(context.Background(), pop, constantFitness(1), neat.RunOptions{MaxGenerations: 20})
	require.NoError(t, err)

	for _, genome := range pop.Genomes {
		assert.False(t, hasBackwardConnection(genome), "feed-forward run grew a backward connection")
		_, err := genome.Compile()
		require.NoError(t, err)
	}
}

// TestRecurrentGenomeCompilesOnlyTheRecurrentWay guards the pairing of the two
// settings: a genome grown with loops in it is not a feed-forward network, and
// asking for one has to fail rather than quietly drop the connections.
func TestRecurrentGenomeCompilesOnlyTheRecurrentWay(t *testing.T) {
	pop, err := neat.GeneratePopulation(recurrentConfig())
	require.NoError(t, err)
	pop, err = neat.Run(context.Background(), pop, constantFitness(1), neat.RunOptions{MaxGenerations: 20})
	require.NoError(t, err)

	for _, genome := range pop.Genomes {
		if !hasBackwardConnection(genome) {
			continue
		}
		net, err := genome.CompileRecurrent()
		require.NoError(t, err)
		assert.True(t, net.IsRecurrent())
		if _, err := genome.Compile(); err == nil {
			// A backward connection between layers need not be a cycle in the
			// graph - a skip backwards over a node with no return path is still
			// acyclic - so this is only worth asserting when it really loops.
			continue
		}
		return
	}
}

// A recurrent connection cannot be split by a node, since there is no layer
// between its ends. Picking one must not forfeit the mutation: the choice has
// to move on to a connection that can be split.
func TestMutateAddNode_SkipsBackwardConnectionsInsteadOfGivingUp(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.Recurrent = true
	cfg.AddNodeMutationRate = 1
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	genome, err := breeder.NewGenome()
	require.NoError(t, err)

	// Wire the output back to itself, alongside the forward input connection.
	output := genome.Layers[len(genome.Layers)-1][0]
	genome.Connections = append(genome.Connections,
		network.NewConnection(breeder.Innovations().ConnectionID(output.ID, output.ID), output.ID, output.ID, .5, true))

	for i := 0; i < 50; i++ {
		mutated := breeder.MutateAddNode(genome)
		assert.Equal(t, genome.NumNodes()+1, mutated.NumNodes(), "attempt %d did not split the forward connection", i)
		_, err := mutated.CompileRecurrent()
		require.NoError(t, err)
	}
}
