package neat

import (
	"github.com/jmwri/neatgo/network"
	"github.com/jmwri/neatgo/util"
)

type Config struct {
	PopulationSize               int
	Layers                       []int
	BiasNodes                    int
	InputActivationFn            network.ActivationFunction
	OutputActivationFn           network.ActivationFunction
	IDProvider                   IDProvider
	RandFloatProvider            util.RandFloatProvider
	MinBias                      float64
	MaxBias                      float64
	MinWeight                    float64
	MaxWeight                    float64
	WeightMutationRate           float64
	WeightFullMutationRate       float64
	AddConnectionMutationRate    float64
	AddNodeMutationRate          float64
	SpeciesCompatExcessCoeff     float64
	SpeciesCompatWeightDiffCoeff float64
	SpeciesCompatThreshold       float64
	SpeciesStalenessThreshold    int
	MateCrossoverRate            float64
	MateBestRate                 float64
}

func DefaultConfig(layers ...int) Config {
	return Config{
		PopulationSize:               100,
		Layers:                       layers,
		BiasNodes:                    1,
		InputActivationFn:            network.ActivationRegistry.Get(network.NoActivation),
		OutputActivationFn:           network.ActivationRegistry.Get(network.Sigmoid),
		IDProvider:                   NewSequentialIDProvider(),
		RandFloatProvider:            util.FloatBetween,
		MinBias:                      -1,
		MaxBias:                      1,
		MinWeight:                    -1,
		MaxWeight:                    1,
		WeightMutationRate:           .8,
		WeightFullMutationRate:       .1,
		AddConnectionMutationRate:    .05,
		AddNodeMutationRate:          .01,
		SpeciesCompatExcessCoeff:     1,
		SpeciesCompatWeightDiffCoeff: 0.5,
		SpeciesCompatThreshold:       3,
		SpeciesStalenessThreshold:    15,
		MateCrossoverRate:            .75,
		MateBestRate:                 .5,
	}
}
