package neat

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/jmwri/neatgo/v2/network"
)

// Unlimited runs every genome in its own goroutine, with no cap on how many
// are in flight at once. Use it for evaluators that spend their time blocked -
// on a simulator, a network call, a subprocess - rather than on the CPU.
const Unlimited = -1

// Evaluator scores a single genome by running its network.
//
// It is called concurrently for many genomes at once, so it must not write to
// shared state without synchronisation. Returning an error aborts the whole
// generation and the error is returned to the caller.
//
// The network is already compiled and may be activated as many times as the
// task needs; a genome that has to be driven through several steps of a game
// simply calls Activate in a loop.
type Evaluator func(ctx context.Context, net *network.Network) (float64, error)

// Evaluate scores every genome in the population, writing the results into
// pop.GenomeFitness.
//
// Genomes are compiled and evaluated in parallel across cfg.Parallelism
// workers. Work is pulled from a shared counter rather than handed out up
// front, so a population of wildly different genome sizes still spreads evenly
// across the workers.
func Evaluate(ctx context.Context, pop Population, eval Evaluator) error {
	if eval == nil {
		return ErrNoEvaluator
	}
	if len(pop.Genomes) == 0 {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	workers := Workers(pop.Cfg.Parallelism, len(pop.Genomes))

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		next     atomic.Int64
		wg       sync.WaitGroup
		failOnce sync.Once
		failure  error
	)
	fail := func(err error) {
		failOnce.Do(func() {
			failure = err
			// Stop the other workers as soon as one of them fails.
			cancel()
		})
	}

	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= len(pop.Genomes) {
					return
				}
				if err := ctx.Err(); err != nil {
					fail(err)
					return
				}

				net, err := pop.Genomes[i].Compile()
				if err != nil {
					fail(fmt.Errorf("%w: genome %d: %w", ErrCompile, i, err))
					return
				}
				fitness, err := eval(ctx, net)
				if err != nil {
					fail(fmt.Errorf("%w: genome %d: %w", ErrEvaluate, i, err))
					return
				}
				pop.GenomeFitness[i] = fitness
			}
		}()
	}
	wg.Wait()

	return failure
}

// Workers resolves a Parallelism setting into a concrete worker count.
func Workers(parallelism, population int) int {
	switch {
	case population <= 0:
		return 0
	case parallelism == Unlimited || parallelism < 0:
		return population
	case parallelism == 0:
		parallelism = runtime.GOMAXPROCS(0)
	}
	if parallelism > population {
		return population
	}
	if parallelism < 1 {
		return 1
	}
	return parallelism
}
