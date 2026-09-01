package neat_test

import (
	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMutateAddNode_NoChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.AddNodeMutationRate = 0
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	genome, err := breeder.NewGenome()
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := breeder.MutateAddNode(genome)
	assert.Equal(t, genome.NumLayers(), actual.NumLayers())
	assert.Equal(t, genome.NumNodes(), actual.NumNodes())
	assert.Equal(t, genome.NumConnections(), actual.NumConnections())
}

func TestMutateAddNode_NodeAdded(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1, 1)
	cfg.AddNodeMutationRate = 1
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
				Type:         network.Output,
				Bias:         1,
				ActivationFn: network.NoActivation,
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
	breeder.Innovations().SetCurrentID(3)

	genome := neat.NewGenome(layers, connections)
	actual := breeder.MutateAddNode(genome)
	assert.Equal(t, genome.NumLayers()+1, actual.NumLayers())
	assert.Equal(t, genome.NumNodes()+1, actual.NumNodes())
	assert.Equal(t, genome.NumConnections()+2, actual.NumConnections())
}
