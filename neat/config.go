package neat

import (
	"github.com/jmwri/neatgo/network"
	"github.com/jmwri/neatgo/util"
)

type Config struct {
	// Providers
	IDProvider        IDProvider
	RandFloatProvider util.RandFloatProvider
	// Number of genomes within a population
	PopulationSize int
	// Number of nodes within each layer
	Layers []int
	// Number of bias nodes
	BiasNodes int
	// Activation functions
	InputActivationFn   network.ActivationFunctionName
	OutputActivationFn  network.ActivationFunctionName
	HiddenActivationFns []network.ActivationFunctionName
	// Node configuration
	AddNodeMutationRate float64
	MinBias             float64
	MaxBias             float64
	// Connection configuration
	MinWeight                 float64
	MaxWeight                 float64
	WeightMutationRate        float64
	WeightMutationPower       float64
	WeightReplaceRate         float64
	AddConnectionMutationRate float64
	// Speciation
	SpeciesCompatExcessCoeff     float64
	SpeciesCompatWeightDiffCoeff float64
	SpeciesCompatThreshold       float64
	SpeciesStalenessThreshold    int
	MateCrossoverRate            float64
	MateBestRate                 float64
}

func DefaultConfig(layers ...int) Config {
	return Config{
		IDProvider:        NewSequentialIDProvider(),
		RandFloatProvider: util.FloatBetween,

		PopulationSize: 100,

		Layers: layers,

		BiasNodes: 1,

		InputActivationFn:   network.NoActivation,
		OutputActivationFn:  network.Sigmoid,
		HiddenActivationFns: network.ActivationRegistry.Names(),

		AddNodeMutationRate: .2,
		MinBias:             -30,
		MaxBias:             30,

		AddConnectionMutationRate: .5,
		MinWeight:                 -30,
		MaxWeight:                 30,
		WeightMutationRate:        .4,
		WeightMutationPower:       .5,
		WeightReplaceRate:         .1,

		SpeciesCompatExcessCoeff:     1,
		SpeciesCompatWeightDiffCoeff: .5,
		SpeciesCompatThreshold:       3,
		SpeciesStalenessThreshold:    15,
		MateCrossoverRate:            .75,
		MateBestRate:                 .5,
	}
}
