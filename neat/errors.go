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
	// ErrNoStopCondition is returned when Run is given options that would
	// never end the run.
	ErrNoStopCondition = errors.New("neat: run would never terminate")
	// ErrCompile is returned when a genome cannot be turned into a network. It
	// wraps the underlying network error.
	ErrCompile = errors.New("neat: genome failed to compile")
	// ErrEvaluate is returned when an evaluator fails. It wraps the error the
	// evaluator returned.
	ErrEvaluate = errors.New("neat: genome failed to evaluate")
)
