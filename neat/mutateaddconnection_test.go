package neat_test

import (
	"github.com/jmwri/neatgo/neat"
	"github.com/jmwri/neatgo/network"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMutateAddConnection_NoChange(t *testing.T) {
	cfg := neat.DefaultGenomeConfig(1, 1)
	cfg.AddConnectionMutationRate = 0
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateAddConnection(cfg, genome)
	assert.Equal(t, genome.NumLayers(), actual.NumLayers())
	assert.Equal(t, genome.NumNodes(), actual.NumNodes())
	assert.Equal(t, genome.NumConnections(), actual.NumConnections())
}

func TestMutateAddConnection_NoChangeWhenFullyConnected(t *testing.T) {
	cfg := neat.DefaultGenomeConfig(1, 1)
	cfg.AddConnectionMutationRate = 1
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateAddConnection(cfg, genome)
	assert.Equal(t, genome.NumLayers(), actual.NumLayers())
	assert.Equal(t, genome.NumNodes(), actual.NumNodes())
	assert.Equal(t, genome.NumConnections(), actual.NumConnections())
}

func TestMutateAddConnection_FullChange(t *testing.T) {
	cfg := neat.DefaultGenomeConfig(1, 2, 2)
	cfg.AddConnectionMutationRate = 1

	layers := [][]network.Node{
		{
			{
				ID:           1,
				Type:         network.Input,
				Bias:         0,
				ActivationFn: network.NoActivationFn,
			},
		},
		{
			{
				ID:           2,
				Type:         network.Hidden,
				Bias:         0,
				ActivationFn: network.NoActivationFn,
			},
			{
				ID:           3,
				Type:         network.Hidden,
				Bias:         0,
				ActivationFn: network.NoActivationFn,
			},
		},
		{
			{
				ID:           4,
				Type:         network.Output,
				Bias:         1,
				ActivationFn: network.NoActivationFn,
			},
			{
				ID:           5,
				Type:         network.Output,
				Bias:         1,
				ActivationFn: network.NoActivationFn,
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

	genome := neat.NewGenome(layers, connections)
	// added1/2 should add connection between nodes 2>5 and 3>4. added3 should have no effect as it is fully connected.
	added1 := neat.MutateAddConnection(cfg, genome)
	added2 := neat.MutateAddConnection(cfg, added1)
	added3 := neat.MutateAddConnection(cfg, added2)
	assert.Equal(t, genome.NumLayers(), added1.NumLayers())
	assert.Equal(t, genome.NumNodes(), added1.NumNodes())
	assert.Equal(t, genome.NumConnections()+1, added1.NumConnections())
	assert.Equal(t, genome.NumLayers(), added2.NumLayers())
	assert.Equal(t, genome.NumNodes(), added2.NumNodes())
	assert.Equal(t, genome.NumConnections()+2, added2.NumConnections())
	assert.Equal(t, genome.NumLayers(), added3.NumLayers())
	assert.Equal(t, genome.NumNodes(), added3.NumNodes())
	assert.Equal(t, genome.NumConnections()+2, added3.NumConnections())
}
