package neat_test

import (
	"github.com/jmwri/neatgo/neat"
	"github.com/jmwri/neatgo/network"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMutateAddNode_NoChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.AddNodeMutationRate = 0
	genome, err := neat.GenerateGenome(cfg)
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := neat.MutateAddNode(cfg, genome)
	assert.Equal(t, genome.NumLayers(), actual.NumLayers())
	assert.Equal(t, genome.NumNodes(), actual.NumNodes())
	assert.Equal(t, genome.NumConnections(), actual.NumConnections())
}

func TestMutateAddNode_NodeAdded(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1, 1)
	cfg.AddNodeMutationRate = 1

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
				Type:         network.Output,
				Bias:         1,
				ActivationFn: network.NoActivationFn,
			},
		},
	}
	connections := []network.Connection{
		{
			ID:      3,
			From:    1,
			To:      2,
			Weight:  .5,
			Enabled: true,
		},
	}
	cfg.IDProvider.SetCurrent(3)

	genome := neat.NewGenome(layers, connections)
	actual := neat.MutateAddNode(cfg, genome)
	assert.Equal(t, genome.NumLayers()+1, actual.NumLayers())
	assert.Equal(t, genome.NumNodes()+1, actual.NumNodes())
	assert.Equal(t, genome.NumConnections()+2, actual.NumConnections())
}
