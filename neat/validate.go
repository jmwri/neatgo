package neat

import (
	"errors"
	"fmt"

	"github.com/jmwri/neatgo/v2/network"
)

// Validate reports everything wrong with a Config.
//
// Bad settings otherwise surface a long way from their cause - a survival
// threshold above 1 slices past the end of a species, a zero population size
// produces an empty run, an unregistered activation function only fails once a
// genome happens to be handed it by a mutation many generations in. All the
// problems are reported together so a config can be fixed in one pass.
func (c Config) Validate() error {
	var problems []error
	bad := func(format string, args ...any) {
		problems = append(problems, fmt.Errorf(format, args...))
	}

	if len(c.Layers) < 2 {
		bad("Layers must have at least an input and an output layer, got %d", len(c.Layers))
	}
	for i, size := range c.Layers {
		if size < 1 {
			bad("Layers[%d] must have at least one node, got %d", i, size)
		}
	}
	if c.PopulationSize < 2 {
		bad("PopulationSize must be at least 2, got %d", c.PopulationSize)
	}
	if c.BiasNodes < 0 {
		bad("BiasNodes must not be negative, got %d", c.BiasNodes)
	}

	rates := map[string]float64{
		"AddNodeMutationRate":          c.AddNodeMutationRate,
		"DeleteNodeMutationRate":       c.DeleteNodeMutationRate,
		"BiasMutationRate":             c.BiasMutationRate,
		"BiasReplaceRate":              c.BiasReplaceRate,
		"ActivationMutationRate":       c.ActivationMutationRate,
		"AddConnectionMutationRate":    c.AddConnectionMutationRate,
		"DeleteConnectionMutationRate": c.DeleteConnectionMutationRate,
		"WeightMutationRate":           c.WeightMutationRate,
		"WeightReplaceRate":            c.WeightReplaceRate,
		"EnabledMutationRate":          c.EnabledMutationRate,
		"MateCrossoverRate":            c.MateCrossoverRate,
		"MateBestRate":                 c.MateBestRate,
		"MateDisabledRate":             c.MateDisabledRate,
	}
	for name, rate := range rates {
		if rate < 0 || rate > 1 {
			bad("%s is a probability and must be between 0 and 1, got %v", name, rate)
		}
	}

	if c.MinWeight > c.MaxWeight {
		bad("MinWeight (%v) must not be greater than MaxWeight (%v)", c.MinWeight, c.MaxWeight)
	}
	if c.MinBias > c.MaxBias {
		bad("MinBias (%v) must not be greater than MaxBias (%v)", c.MinBias, c.MaxBias)
	}
	spreads := map[string]float64{
		"WeightInitStdDev":    c.WeightInitStdDev,
		"WeightMutationPower": c.WeightMutationPower,
		"BiasInitStdDev":      c.BiasInitStdDev,
		"BiasMutationPower":   c.BiasMutationPower,
	}
	for name, spread := range spreads {
		if spread < 0 {
			bad("%s must not be negative, got %v", name, spread)
		}
	}

	if c.SurvivalThreshold <= 0 || c.SurvivalThreshold > 1 {
		bad("SurvivalThreshold is a fraction of each species and must be above 0 and at most 1, got %v", c.SurvivalThreshold)
	}
	if c.SpeciesCompatThreshold <= 0 {
		bad("SpeciesCompatThreshold must be above 0, got %v", c.SpeciesCompatThreshold)
	}
	if c.MinSpeciesCompatThreshold < 0 {
		bad("MinSpeciesCompatThreshold must not be negative, got %v", c.MinSpeciesCompatThreshold)
	}
	if c.SpeciesElitism < 0 {
		bad("SpeciesElitism must not be negative, got %d", c.SpeciesElitism)
	}
	if c.SpeciesStalenessThreshold < 1 {
		bad("SpeciesStalenessThreshold must be at least 1, got %d", c.SpeciesStalenessThreshold)
	}
	if c.Elitism < 0 {
		bad("Elitism must not be negative, got %d", c.Elitism)
	}
	if c.MinSpeciesSize < 1 {
		bad("MinSpeciesSize must be at least 1, got %d", c.MinSpeciesSize)
	}
	if c.Elitism > c.PopulationSize {
		bad("Elitism (%d) must not exceed PopulationSize (%d)", c.Elitism, c.PopulationSize)
	}
	if c.Parallelism < Unlimited {
		bad("Parallelism must be Unlimited, 0, or a positive worker count, got %d", c.Parallelism)
	}

	activations := map[string]network.ActivationFunctionName{
		"InputActivationFn":  c.InputActivationFn,
		"OutputActivationFn": c.OutputActivationFn,
	}
	for name, fn := range activations {
		if network.ActivationRegistry.Get(fn) == nil {
			bad("%s %q is not registered", name, fn)
		}
	}
	if len(c.HiddenActivationFns) == 0 {
		bad("HiddenActivationFns must list at least one activation function")
	}
	for i, fn := range c.HiddenActivationFns {
		if network.ActivationRegistry.Get(fn) == nil {
			bad("HiddenActivationFns[%d] %q is not registered", i, fn)
		}
	}

	if len(problems) == 0 {
		return nil
	}
	return fmt.Errorf("%w: %w", ErrInvalidConfig, errors.Join(problems...))
}
