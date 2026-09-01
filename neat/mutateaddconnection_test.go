package neat_test

import (
	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMutateAddConnection_NoChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.AddConnectionMutationRate = 0
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	genome, err := breeder.NewGenome()
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := breeder.MutateAddConnection(genome)
	assert.Equal(t, genome.NumLayers(), actual.NumLayers())
	assert.Equal(t, genome.NumNodes(), actual.NumNodes())
	assert.Equal(t, genome.NumConnections(), actual.NumConnections())
}

func TestMutateAddConnection_NoChangeWhenFullyConnected(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.AddConnectionMutationRate = 1
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	genome, err := breeder.NewGenome()
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := breeder.MutateAddConnection(genome)
	assert.Equal(t, genome.NumLayers(), actual.NumLayers())
	assert.Equal(t, genome.NumNodes(), actual.NumNodes())
	assert.Equal(t, genome.NumConnections(), actual.NumConnections())
}

func TestMutateAddConnection_FullChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 2, 2)
	cfg.AddConnectionMutationRate = 1
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)

	layers := [][]network.Node{
		{
			{
				ID:           1,
				Type:         network.Input,
				Bias:         0,
				ActivationFn: network.NoActivation,
			},
		},
		{
			{
				ID:           2,
				Type:         network.Hidden,
				Bias:         0,
				ActivationFn: network.NoActivation,
			},
			{
				ID:           3,
				Type:         network.Hidden,
				Bias:         0,
				ActivationFn: network.NoActivation,
			},
		},
		{
			{
				ID:           4,
				Type:         network.Output,
				Bias:         1,
				ActivationFn: network.NoActivation,
			},
			{
				ID:           5,
				Type:         network.Output,
				Bias:         1,
				ActivationFn: network.NoActivation,
			},
		},
	}
	connections := []network.Connection{
		{
			ID:      6,
			From:    1,
			To:      2,
			Weight:  .8,
			Enabled: true,
		},
		{
			ID:      7,
			From:    1,
			To:      3,
			Weight:  .5,
			Enabled: true,
		},
		{
			ID:      8,
			From:    2,
			To:      4,
			Weight:  1,
			Enabled: true,
		},
		{
			ID:      9,
			From:    3,
			To:      5,
			Weight:  .5,
			Enabled: true,
		},
	}
	breeder.Innovations().SetCurrentID(9)

	genome := neat.NewGenome(layers, connections)
	// Four connections are missing and may be added without creating a cycle:
	// the skip connections 1>4 and 1>5, plus 2>5 and 3>4. Once all four exist
	// the genome is saturated and further mutations are no-ops.
	added := genome
	for i := 1; i <= 4; i++ {
		added = breeder.MutateAddConnection(added)
		assert.Equal(t, genome.NumLayers(), added.NumLayers())
		assert.Equal(t, genome.NumNodes(), added.NumNodes())
		assert.Equal(t, genome.NumConnections()+i, added.NumConnections())
	}

	saturated := breeder.MutateAddConnection(added)
	assert.Equal(t, added.NumConnections(), saturated.NumConnections(), "no connection left to add")
}
