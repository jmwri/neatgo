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
	HiddenActivationFns []network.ActivationFunctionName // Activation functions available for hidden nodes. Default is all of them.
	// Node configuration
	AddNodeMutationRate    float64 // How often to add a node.
	DeleteNodeMutationRate float64 // How often to delete a node.
	MinBias                float64 // Min node bias.
	MaxBias                float64 // Max node bias.
	BiasMutationRate       float64 // How often to mutate nodes bias.
	BiasMutationPower      float64 // How much to mutate the bias. Calculated as node.bias +/- (node.bias*power).
	BiasReplaceRate        float64 // How often to create a completely new bias, instead of mutating the existing one.
	// Connection configuration
	AddConnectionMutationRate float64 // How often to add a connection.
	MinWeight                 float64 // Min connection weight.
	MaxWeight                 float64 // Max connection weight.
	WeightMutationRate        float64 // How often to mutate connection weight.
	WeightMutationPower       float64 // How much to mutate the weight. Calculated as connection.weight +/- (connection.weight*power).
	WeightReplaceRate         float64 // How often to create a completely new weight, instead of mutating the existing one.
	// Speciation
	SpeciesCompatExcessCoeff     float64 // How important are disjoint + excess genes when calculating species?
	SpeciesCompatBiasDiffCoeff   float64 // How important are node biases when calculating species?
	SpeciesCompatWeightDiffCoeff float64 // How important are connection weights when calculating species?
	SpeciesCompatThreshold       float64 // How similar should genomes be to be considered the same species? Lower = more similar.
	SpeciesStalenessThreshold    int     // If species does not improve after this many generations it will be removed.
	// Crossover
	MateCrossoverRate float64 // How often to perform crossover between 2 parents in same species. Otherwise, take a random genome in the species.
	MateBestRate      float64 // How often should we take the gene from the best genome.
	// Population
	TopGenomesFromSpeciesToFill int // How many top genomes to take from each species to fill any remaining population.
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
		BiasMutationRate:    .8,
		BiasMutationPower:   .2,
		BiasReplaceRate:     .01,

		AddConnectionMutationRate: .8,
		MinWeight:                 -30,
		MaxWeight:                 30,
		WeightMutationRate:        .8,
		WeightMutationPower:       .2,
		WeightReplaceRate:         .01,

		SpeciesCompatExcessCoeff:     1,
		SpeciesCompatBiasDiffCoeff:   .5,
		SpeciesCompatWeightDiffCoeff: .5,
		SpeciesCompatThreshold:       5,
		SpeciesStalenessThreshold:    15,

		MateCrossoverRate: .75,
		MateBestRate:      .8,

		TopGenomesFromSpeciesToFill: 2,
	}
}
