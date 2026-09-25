package neat

import "errors"

// Errors returned by the evolutionary loop.
//
// These are distinguishable with errors.Is. An evaluator's own error is
// wrapped rather than replaced, so errors.Is and errors.As still find it.
var (
	// ErrInvalidConfig is returned by Validate and by GeneratePopulation when
	// a setting is out of range. It wraps one error per problem found.
	ErrInvalidConfig = errors.New("neat: invalid config")
	// ErrNoEvaluator is returned when Evaluate is given a nil evaluator.
	ErrNoEvaluator = errors.New("neat: evaluator must not be nil")
	// ErrPopulationShape is returned when a Population's slices do not line
	// up, such as a fitness slice that is not one entry per genome.
	ErrPopulationShape = errors.New("neat: population slices do not match")
	// ErrNoStopCondition is returned when Run is given options that would
	// never end the run.
	ErrNoStopCondition = errors.New("neat: run would never terminate")
	// ErrCompile is returned when a genome cannot be turned into a network. It
	// wraps the underlying network error.
	ErrCompile = errors.New("neat: genome failed to compile")
	// ErrEvaluate is returned when an evaluator fails. It wraps the error the
	// evaluator returned.
	ErrEvaluate = errors.New("neat: genome failed to evaluate")
	// ErrBatch is returned when the Activator of a Batch fails or returns
	// something that does not match what it was asked for. It wraps the
	// activator's own error.
	ErrBatch = errors.New("neat: batch activation failed")
	// ErrEmptyBatch is returned when a Batch has no inputs to run.
	ErrEmptyBatch = errors.New("neat: batch has nothing to run")
	// ErrRecurrentBatch is returned when a Batch is used with a recurrent
	// population. A batch treats its samples as independent, which a network that
	// remembers between activations contradicts.
	ErrRecurrentBatch = errors.New("neat: a batch cannot evaluate a recurrent population")
)
