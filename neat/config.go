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
	HiddenActivationFns          []network.ActivationFunctionName
	IDProvider                   IDProvider
	RandFloatProvider            util.RandFloatProvider
	MinBias                      float64
	MaxBias                      float64
	MinWeight                    float64
	MaxWeight                    float64
	WeightMutationRate           float64
	WeightMutationPower          float64
	WeightReplaceRate            float64
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
		HiddenActivationFns:          network.ActivationRegistry.Names(),
		IDProvider:                   NewSequentialIDProvider(),
		RandFloatProvider:            util.FloatBetween,
		MinBias:                      -30,
		MaxBias:                      30,
		MinWeight:                    -30,
		MaxWeight:                    30,
		WeightMutationRate:           .4,
		WeightMutationPower:          .5,
		WeightReplaceRate:            .1,
		AddConnectionMutationRate:    .5,
		AddNodeMutationRate:          .2,
		SpeciesCompatExcessCoeff:     2,
		SpeciesCompatWeightDiffCoeff: 2,
		SpeciesCompatThreshold:       3,
		SpeciesStalenessThreshold:    15,
		MateCrossoverRate:            .75,
		MateBestRate:                 .5,
	}
}
