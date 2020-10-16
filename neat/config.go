package neat

import (
	"github.com/jmwri/neatgo/activation"
	"github.com/jmwri/neatgo/aggregation"
)

func NewConfig(popSize int, numInputs int, numOutputs int) Config {
	return Config{
		FitnessCriterion:                 aggregation.Max,
		FitnessThreshold:                 0,
		NoFitnessTermination:             true,
		PopulationSize:                   popSize,
		ResetOnExtinction:                false,
		SpeciesFitnessFn:                 aggregation.Mean,
		MaxStagnation:                    15,
		SpeciesElitism:                   2,
		Elitism:                          2,
		SurvivalThreshold:                0.2,
		MinSpeciesSize:                   2,
		FitnessMinDivisor:                1,
		ActivationDefault:                activation.Sigmoid,
		ActivationMutateRate:             .05,
		ActivationOptions:                activation.FnAll,
		AggregationDefault:               aggregation.Sum,
		AggregationMutateRate:            .05,
		AggregationOptions:               aggregation.FnAll,
		BiasInitMax:                      1,
		BiasInitMin:                      0,
		BiasMaxValue:                     1,
		BiasMinValue:                     0,
		BiasMutatePower:                  .02,
		BiasMutateRate:                   .7,
		BiasReplaceRate:                  .1,
		CompatibilityThreshold:           3,
		CompatibilityDisjointCoefficient: 1,
		CompatibilityWeightCoefficient:   .5,
		ConnAddProb:                      .5,
		ConnDeleteProb:                   .5,
		EnabledDefault:                   true,
		EnabledMutateRate:                .01,
		EnabledRateToFalseAdd:            .1,
		EnabledRateToTrueAdd:             .25,
		NodeAddProb:                      .1,
		NodeDeleteProb:                   .05,
		NumHidden:                        []int{},
		NumInputs:                        numInputs,
		NumOutputs:                       numOutputs,
		SingleStructuralMutation:         false,
		StructuralMutationSurer:          false,
		WeightInitMax:                    1,
		WeightInitMin:                    0,
		WeightMaxValue:                   1,
		WeightMinValue:                   0,
		WeightMutatePower:                0.02,
		WeightMutateRate:                 0.2,
		WeightReplaceRate:                0.1,
	}
}

type Config struct {
	// The NEAT section specifies parameters particular to the generic NEAT algorithm or the experiment itself. This section is always required, and is handled by the Config class itself.

	// The function used to compute the termination criterion from the set of genome fitnesses. Allowable values are: min, max, and mean
	FitnessCriterion aggregation.Fn
	// When the fitness computed by fitness_criterion meets or exceeds this threshold, the evolution process will terminate, with a call to any registered reporting class’ found_solution method.
	FitnessThreshold float64
	// If this evaluates to True, then the fitness_criterion and fitness_threshold are ignored for termination; only valid if termination by a maximum number of generations passed to population.Population.run() is enabled, and the found_solution method is called upon generation number termination. If it evaluates to False, then fitness is used to determine termination. This defaults to “False”.
	NoFitnessTermination bool
	// The number of individuals in each generation.
	PopulationSize int
	// If this evaluates to True, when all species simultaneously become extinct due to stagnation, a new random population will be created. If False, a CompleteExtinctionException will be thrown.
	ResetOnExtinction bool

	// The DefaultStagnation section specifies parameters for the builtin DefaultStagnation class. This section is only necessary if you specify this class as the stagnation implementation when creating the Config instance; otherwise you need to include whatever configuration (if any) is required for your particular implementation.

	// The function used to compute species fitness. This defaults to ``mean``. Allowed values are: max, min, mean, and median
	SpeciesFitnessFn aggregation.Fn
	// Species that have not shown improvement in more than this number of generations will be considered stagnant and removed. This defaults to 15.
	MaxStagnation int
	// The number of species that will be protected from stagnation; mainly intended to prevent total extinctions caused by all species becoming stagnant before new species arise. For example, a species_elitism setting of 3 will prevent the 3 species with the highest species fitness from being removed for stagnation regardless of the amount of time they have not shown improvement. This defaults to 0.
	SpeciesElitism int

	// The DefaultReproduction section specifies parameters for the builtin DefaultReproduction class. This section is only necessary if you specify this class as the reproduction implementation when creating the Config instance; otherwise you need to include whatever configuration (if any) is required for your particular implementation.

	// The number of most-fit individuals in each species that will be preserved as-is from one generation to the next. This defaults to 0.
	Elitism int
	// The fraction for each species allowed to reproduce each generation. This defaults to 0.2.
	SurvivalThreshold float64
	// The minimum number of genomes per species after reproduction. This defaults to 2.
	MinSpeciesSize int
	// The min divisor to be used when generating the fitness range of a species set.
	FitnessMinDivisor float64

	// The DefaultGenome section specifies parameters for the builtin DefaultGenome class. This section is only necessary if you specify this class as the genome implementation when creating the Config instance; otherwise you need to include whatever configuration (if any) is required for your particular implementation.

	// The default activation function attribute assigned to new nodes. If none is given, or “random” is specified, one of the activation_options will be chosen at random.
	ActivationDefault activation.Fn
	// The probability that mutation will replace the node’s activation function with a randomly-determined member of the activation_options. Valid values are in [0.0, 1.0].
	ActivationMutateRate float64
	// A space-separated list of the activation functions that may be used by nodes. This defaults to sigmoid. The built-in available functions can be found in Overview of builtin activation functions; more can be added as described in Customizing Behavior.
	ActivationOptions []activation.Fn
	// The default aggregation function attribute assigned to new nodes. If none is given, or “random” is specified, one of the aggregation_options will be chosen at random.
	AggregationDefault aggregation.Fn
	// The probability that mutation will replace the node’s aggregation function with a randomly-determined member of the aggregation_options. Valid values are in [0.0, 1.0].
	AggregationMutateRate float64
	// A space-separated list of the aggregation functions that may be used by nodes. This defaults to “sum”. The available functions (defined in aggregations) are: sum, product, min, max, mean, median, and maxabs (which returns the input value with the greatest absolute value; the returned value may be positive or negative). New aggregation functions can be defined similarly to new activation functions. (Note that the function needs to take a list or other iterable; the reduce function, as in aggregations, may be of use in this.)
	AggregationOptions []aggregation.Fn
	// The maximum init value of bias
	BiasInitMax float64
	// The minimum init value of bias
	BiasInitMin float64
	// The maximum allowed bias value. Biases above this value will be clamped to this value.
	BiasMaxValue float64
	// The minimum allowed bias value. Biases below this value will be clamped to this value.
	BiasMinValue float64
	// The standard deviation of the zero-centered normal/gaussian distribution from which a bias value mutation is drawn.
	BiasMutatePower float64
	// The probability that mutation will change the bias of a node by adding a random value.
	BiasMutateRate float64
	// The probability that mutation will replace the bias of a node with a newly chosen random value (as if it were a new node).
	BiasReplaceRate float64
	// Individuals whose genomic distance is less than this threshold are considered to be in the same species.
	CompatibilityThreshold float64
	// The coefficient for the disjoint and excess gene counts’ contribution to the genomic distance.
	CompatibilityDisjointCoefficient float64
	// The coefficient for each weight, bias, or response multiplier difference’s contribution to the genomic distance (for homologous nodes or connections). This is also used as the value to add for differences in activation functions, aggregation functions, or enabled/disabled status.
	CompatibilityWeightCoefficient float64
	// The probability that mutation will add a connection between existing nodes. Valid values are in [0.0, 1.0].
	ConnAddProb float64
	// The probability that mutation will delete an existing connection. Valid values are in [0.0, 1.0].
	ConnDeleteProb float64
	// The default enabled attribute of newly created connections. Valid values are True and False.
	EnabledDefault bool
	// The probability that mutation will replace (50/50 chance of True or False) the enabled status of a connection. Valid values are in [0.0, 1.0].
	EnabledMutateRate float64
	// Adds to the enabled_mutate_rate if the connection is currently enabled.
	EnabledRateToFalseAdd float64
	// Adds to the enabled_mutate_rate if the connection is currently not enabled.
	EnabledRateToTrueAdd float64
	// The probability that mutation will add a new node (essentially replacing an existing connection, the enabled status of which will be set to False). Valid values are in [0.0, 1.0].
	NodeAddProb float64
	// The probability that mutation will delete an existing node (and all connections to it). Valid values are in [0.0, 1.0].
	NodeDeleteProb float64
	// The number of hidden nodes in each hidden layer to add to each genome in the initial population.
	NumHidden []int
	// The number of input nodes, through which the network receives inputs.
	NumInputs int
	// The number of output nodes, to which the network delivers outputs.
	NumOutputs int
	// If this evaluates to True, only one structural mutation (the addition or removal of a node or connection) will be allowed per genome per generation. (If the probabilities for conn_add_prob, conn_delete_prob, node_add_prob, and node_delete_prob add up to over 1, the chances of each are proportional to the appropriate configuration value.) This defaults to “False”.
	SingleStructuralMutation bool
	// If this evaluates to True, then an attempt to add a node to a genome lacking connections will result in adding a connection instead; furthermore, if an attempt to add a connection tries to add a connection that already exists, that connection will be enabled. If this is set to default, then it acts as if it had the same value as single_structural_mutation (above). This defaults to “default”.
	StructuralMutationSurer bool
	// The maximum init value of weight
	WeightInitMax float64
	// The minimum init value of weight
	WeightInitMin float64
	// The maximum allowed weight value. Weights above this value will be clamped to this value.
	WeightMaxValue float64
	// The minimum allowed weight value. Weights below this value will be clamped to this value.
	WeightMinValue float64
	// The standard deviation of the zero-centered normal/gaussian distribution from which a weight value mutation is drawn.
	WeightMutatePower float64
	// The probability that mutation will change the weight of a connection by adding a random value.
	WeightMutateRate float64
	// The probability that mutation will replace the weight of a connection with a newly chosen random value (as if it were a new connection).
	WeightReplaceRate float64
}
