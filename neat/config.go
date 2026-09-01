package neat

import (
	"github.com/jmwri/neatgo/v2/internal/util"
	"github.com/jmwri/neatgo/v2/network"
)

type Config struct {
	// Seed fixes the random sequence for a run, making it reproducible. Zero
	// draws a fresh seed, which GeneratePopulation records on the population so
	// an interesting run can be replayed later.
	Seed uint64
	// Number of genomes within a population
	PopulationSize int
	// Number of nodes within each layer
	Layers []int
	// Number of bias nodes
	BiasNodes int
	// Activation functions
	InputActivationFn   network.ActivationFunctionName
	OutputActivationFn  network.ActivationFunctionName
	HiddenActivationFns []network.ActivationFunctionName // Activation functions available for hidden nodes.
	// Node configuration
	AddNodeMutationRate    float64 // How often to add a node.
	DeleteNodeMutationRate float64 // How often to delete a node.
	MinBias                float64 // Min node bias.
	MaxBias                float64 // Max node bias.
	BiasInitStdDev         float64 // Std dev of the gaussian a new bias is drawn from.
	BiasMutationRate       float64 // How often to mutate a node's bias.
	BiasMutationPower      float64 // Std dev of the gaussian added to a bias when perturbing it.
	BiasReplaceRate        float64 // How often to draw a completely new bias, instead of perturbing the existing one.
	ActivationMutationRate float64 // How often to mutate a hidden node's activation function.
	// Connection configuration
	AddConnectionMutationRate    float64 // How often to add a connection.
	DeleteConnectionMutationRate float64 // How often to delete a connection.
	MinWeight                    float64 // Min connection weight.
	MaxWeight                    float64 // Max connection weight.
	WeightInitStdDev             float64 // Std dev of the gaussian a new weight is drawn from.
	WeightMutationRate           float64 // How often to mutate a connection's weight.
	WeightMutationPower          float64 // Std dev of the gaussian added to a weight when perturbing it.
	WeightReplaceRate            float64 // How often to draw a completely new weight, instead of perturbing the existing one.
	EnabledMutationRate          float64 // How often to flip a connection's enabled flag.
	// Speciation
	SpeciesElitism               int     // The number of top species to protect from stagnation.
	SpeciesCompatExcessCoeff     float64 // How important are disjoint + excess genes when calculating species?
	SpeciesCompatBiasDiffCoeff   float64 // How important are node biases when calculating species?
	SpeciesCompatWeightDiffCoeff float64 // How important are connection weights when calculating species?
	SpeciesCompatThreshold       float64 // How similar should genomes be to be considered the same species? Lower = more similar.
	SpeciesStalenessThreshold    int     // If species does not improve after this many generations it will be removed.
	TargetSpecies                int     // Species count to steer SpeciesCompatThreshold towards each generation. 0 disables the adjustment.
	SpeciesCompatThresholdAdjust float64 // How far to move SpeciesCompatThreshold per generation when off target.
	MinSpeciesCompatThreshold    float64 // Floor for SpeciesCompatThreshold, so it can never collapse to a single species.
	// Crossover
	SurvivalThreshold float64 // The fraction of each species to allow for reproduction.
	MateCrossoverRate float64 // How often to perform crossover between 2 parents in same species. Otherwise, take a random genome in the species.
	MateBestRate      float64 // For a gene held by both parents, how often to take the fitter parent's copy.
	MateDisabledRate  float64 // If a gene is disabled in either parent, how often it stays disabled in the child.
	// Population
	Elitism        int // How many top genomes to take from each species without mutation.
	MinSpeciesSize int // Minimum number of offspring allocated to a surviving species.
	// Parallelism caps how many genomes are evaluated concurrently.
	//	 0         one worker per CPU (GOMAXPROCS). The default, and the right
	//	           choice for evaluators that are CPU bound.
	//	 n > 0     exactly n workers.
	//	 Unlimited one goroutine per genome, for evaluators that spend their
	//	           time blocked rather than computing.
	Parallelism int
}

// Chance reports whether an event with probability p occurs.
func (c Config) Chance(rng *Rand, p float64) bool {
	return util.Chance(rng, p)
}

// RandWeight draws a fresh connection weight.
func (c Config) RandWeight(rng *Rand) float64 {
	return util.Clamp(util.Gaussian(rng)*c.WeightInitStdDev, c.MinWeight, c.MaxWeight)
}

// PerturbWeight nudges an existing connection weight.
func (c Config) PerturbWeight(rng *Rand, weight float64) float64 {
	return util.Clamp(weight+util.Gaussian(rng)*c.WeightMutationPower, c.MinWeight, c.MaxWeight)
}

// RandBias draws a fresh node bias.
func (c Config) RandBias(rng *Rand) float64 {
	return util.Clamp(util.Gaussian(rng)*c.BiasInitStdDev, c.MinBias, c.MaxBias)
}

// PerturbBias nudges an existing node bias.
func (c Config) PerturbBias(rng *Rand, bias float64) float64 {
	return util.Clamp(bias+util.Gaussian(rng)*c.BiasMutationPower, c.MinBias, c.MaxBias)
}

func DefaultConfig(layers ...int) Config {
	return Config{
		PopulationSize: 150,

		Layers: layers,

		BiasNodes: 1,

		InputActivationFn:  network.NoActivation,
		OutputActivationFn: network.Sigmoid,
		// A small, well behaved default. Handing hidden nodes the whole
		// registry (exp, inv, log, cube, ...) makes the search wander through
		// wildly scaled functions and is rarely what you want.
		HiddenActivationFns: []network.ActivationFunctionName{
			network.Sigmoid,
			network.Tanh,
			network.Relu,
		},

		// Structural mutations are rare on purpose. NEAT's whole premise is
		// that topology grows slowly from a minimal start, so that a new
		// structure has generations to prove itself before the next one lands.
		// Deletion is rarer than addition, otherwise structure erodes as fast
		// as it appears.
		AddNodeMutationRate:    .03,
		DeleteNodeMutationRate: .01,
		MinBias:                -8,
		MaxBias:                8,
		BiasInitStdDev:         1,
		BiasMutationRate:       .7,
		BiasMutationPower:      .5,
		BiasReplaceRate:        .1,
		ActivationMutationRate: 0,

		AddConnectionMutationRate:    .08,
		DeleteConnectionMutationRate: .02,
		MinWeight:                    -8,
		MaxWeight:                    8,
		WeightInitStdDev:             1,
		WeightMutationRate:           .8,
		WeightMutationPower:          .5,
		WeightReplaceRate:            .1,
		EnabledMutationRate:          .01,

		SpeciesElitism:               2,
		SpeciesCompatExcessCoeff:     1,
		SpeciesCompatBiasDiffCoeff:   .5,
		SpeciesCompatWeightDiffCoeff: .5,
		SpeciesCompatThreshold:       3,
		SpeciesStalenessThreshold:    20,
		TargetSpecies:                10,
		SpeciesCompatThresholdAdjust: .2,
		MinSpeciesCompatThreshold:    .5,

		SurvivalThreshold: .2,
		MateCrossoverRate: .75,
		MateBestRate:      .5,
		MateDisabledRate:  .75,

		Elitism:        2,
		MinSpeciesSize: 2,
		Parallelism:    0,
	}
}
