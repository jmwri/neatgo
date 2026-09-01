package neat_test

import (
	"fmt"
	"github.com/jmwri/neatgo/v2/neat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMutateNodeBiases_NoChange(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.BiasNodes = 0
	cfg.BiasMutationRate = 0
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	genome, err := breeder.NewGenome()
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := breeder.MutateNodeBiases(genome)
	assert.Equal(t, fmt.Sprint(genome), fmt.Sprint(actual))
}

func TestMutateNodeBiases_FullMutation(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.BiasNodes = 0
	cfg.BiasMutationRate = 1
	cfg.BiasReplaceRate = 1
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	genome, err := breeder.NewGenome()
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := breeder.MutateNodeBiases(genome)
	assert.NotEqual(t, fmt.Sprint(genome), fmt.Sprint(actual))
}

func TestMutateNodeBiases_MinimalMutation(t *testing.T) {
	cfg := neat.DefaultConfig(1, 1)
	cfg.BiasNodes = 0
	cfg.BiasMutationRate = 1
	cfg.BiasReplaceRate = 0
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)
	genome, err := breeder.NewGenome()
	assert.NoError(t, err, "unexpected error when generating genome")
	actual := breeder.MutateNodeBiases(genome)
	assert.NotEqual(t, fmt.Sprint(genome), fmt.Sprint(actual))
}
