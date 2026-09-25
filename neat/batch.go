package neat

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/jmwri/neatgo/v2/network"
)

// Activator runs many networks over many inputs at once. The gpu package's
// Device implements it; CPUActivator is the same thing on the CPU.
//
// The result holds, for every network, one output vector per input sample. Every
// network is feed-forward, and the samples are independent of one another.
type Activator interface {
	ActivateBatch(nets []*network.Network, inputs [][]float64) (network.BatchOutputs, error)
}

// CPUActivator is an Activator that runs on the CPU, across Parallelism
// goroutines (0 means one per CPU). Use it where no GPU is available, and to
// check what a GPU run should be producing.
type CPUActivator struct {
	Parallelism int
}

// ActivateBatch implements Activator.
func (c CPUActivator) ActivateBatch(nets []*network.Network, inputs [][]float64) (network.BatchOutputs, error) {
	return network.ActivateBatch(nets, inputs, c.Parallelism)
}

// BatchScorer turns one network's outputs over the whole dataset into a
// fitness. It is called concurrently for many genomes, like an Evaluator, and
// must not write to shared state without synchronisation.
//
// net is the network that produced the outputs, for a score that depends on its
// size as well as its answers - a complexity penalty, say.
type BatchScorer func(net *network.Network, outputs network.Outputs) (float64, error)

// Batch describes an evaluation that is the same computation for every genome:
// run the network over a fixed dataset and score what comes out. That shape is
// what lets the whole population be activated in one go, on a GPU if the
// Activator is one.
//
// It does not fit a task where the input depends on the previous output, as in
// a game; use an Evaluator for those.
type Batch struct {
	// Activator runs the population over Inputs.
	Activator Activator
	// Inputs is the dataset: one input vector per sample, each the size the
	// genomes' input layer expects.
	Inputs [][]float64
	// Score reduces a genome's outputs to its fitness.
	Score BatchScorer
}

// EvaluateBatch scores every genome in the population, writing the results into
// pop.GenomeFitness, by running the whole population over b.Inputs with
// b.Activator and scoring each genome's outputs.
//
// It is Evaluate for a Batch. Genomes are compiled and scored in parallel across
// cfg.Parallelism workers; the activation in between is the Activator's.
// Recurrent populations are rejected, as a batch treats samples as independent
// and a recurrent network's answer depends on what came before.
func EvaluateBatch(ctx context.Context, pop Population, b Batch) error {
	if b.Activator == nil || b.Score == nil {
		return ErrNoEvaluator
	}
	if len(b.Inputs) == 0 {
		return fmt.Errorf("%w: no inputs", ErrEmptyBatch)
	}
	if pop.Cfg.Recurrent {
		return ErrRecurrentBatch
	}
	if len(pop.Genomes) == 0 {
		return nil
	}
	if len(pop.GenomeFitness) != len(pop.Genomes) {
		return fmt.Errorf("%w: %d genomes but %d fitness slots", ErrPopulationShape, len(pop.Genomes), len(pop.GenomeFitness))
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	workers := Workers(pop.Cfg.Parallelism, len(pop.Genomes))

	nets := make([]*network.Network, len(pop.Genomes))
	if err := parallelEach(ctx, len(nets), workers, func(i int) error {
		net, err := pop.Genomes[i].CompileFor(pop.Cfg)
		if err != nil {
			return fmt.Errorf("%w: genome %d: %w", ErrCompile, i, err)
		}
		nets[i] = net
		return nil
	}); err != nil {
		return err
	}

	outputs, err := b.Activator.ActivateBatch(nets, b.Inputs)
	if err != nil {
		return fmt.Errorf("%w: %w", ErrBatch, err)
	}
	if outputs.NumNets != len(nets) || outputs.NumSamples != len(b.Inputs) {
		return fmt.Errorf("%w: activator returned %d networks by %d samples, expected %d by %d",
			ErrBatch, outputs.NumNets, outputs.NumSamples, len(nets), len(b.Inputs))
	}

	return parallelEach(ctx, len(nets), workers, func(i int) error {
		fitness, err := b.Score(nets[i], outputs.Net(i))
		if err != nil {
			return fmt.Errorf("%w: genome %d: %w", ErrEvaluate, i, err)
		}
		pop.GenomeFitness[i] = fitness
		return nil
	})
}

// RunGenerationBatch is RunGeneration for a Batch: it evaluates the whole
// population with EvaluateBatch and returns the next generation. On error the
// population comes back without having advanced.
func RunGenerationBatch(ctx context.Context, pop Population, b Batch) (Population, error) {
	if err := EvaluateBatch(ctx, pop, b); err != nil {
		return pop, err
	}
	return Advance(pop), nil
}

// RunBatch is Run for a Batch. It ends the same ways Run does, and likewise
// always returns the population.
func RunBatch(ctx context.Context, pop Population, b Batch, opts RunOptions) (Population, error) {
	return run(ctx, pop, opts, func(pop Population) (Population, error) {
		return RunGenerationBatch(ctx, pop, b)
	})
}

// parallelEach calls fn for every index in [0, n) across the given number of
// workers, pulling indices from a shared counter so uneven work still spreads
// evenly. The first error stops the rest and is returned.
func parallelEach(ctx context.Context, n, workers int, fn func(i int) error) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		next     atomic.Int64
		wg       sync.WaitGroup
		failOnce sync.Once
		failure  error
	)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= n {
					return
				}
				err := ctx.Err()
				if err == nil {
					err = fn(i)
				}
				if err != nil {
					failOnce.Do(func() {
						failure = err
						cancel()
					})
					return
				}
			}
		}()
	}
	wg.Wait()
	return failure
}
