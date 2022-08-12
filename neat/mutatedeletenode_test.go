package neat_test

import (
	"github.com/jmwri/neatgo/neat"
	"github.com/jmwri/neatgo/network"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMutateDeleteNode_NoChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.DeleteNodeMutationRate = 0
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateDeleteNode(cfg, genome)
	assert.Equal(t, genome.NumLayers(), actual.NumLayers())
	assert.Equal(t, genome.NumNodes(), actual.NumNodes())
	assert.Equal(t, genome.NumConnections(), actual.NumConnections())
}

func TestMutateDeleteNode_NodeDeleted(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1, 1)
	cfg.DeleteNodeMutationRate = 1

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
		},
		{
			{
				ID:           3,
				Type:         network.Output,
				Bias:         1,
				ActivationFn: network.NoActivation,
			},
		},
	}
	connections := []network.Connection{
		{
			ID:      4,
			From:    1,
			To:      2,
			Weight:  .5,
			Enabled: true,
		},
		{
			ID:      5,
			From:    2,
			To:      3,
			Weight:  .5,
			Enabled: true,
		},
	}
	cfg.IDProvider.SetCurrent(3)

	genome := neat.NewGenome(layers, connections)
	actual := neat.MutateDeleteNode(cfg, genome)
	assert.Equal(t, genome.NumLayers()-1, actual.NumLayers())
	assert.Equal(t, genome.NumNodes()-1, actual.NumNodes())
	assert.Equal(t, genome.NumConnections()-2, actual.NumConnections())
}
