package neat_test

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testPopulation(t *testing.T, size int) neat.Population {
	t.Helper()
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = size
	pop, err := neat.GeneratePopulation(cfg)
	require.NoError(t, err)
	return pop
}

// peakConcurrency returns an evaluator that records the highest number of
// evaluations running at the same moment.
func peakConcurrency(hold time.Duration) (neat.Evaluator, *atomic.Int64) {
	var inFlight, peak atomic.Int64
	eval := func(context.Context, *network.Network) (float64, error) {
		current := inFlight.Add(1)
		for {
			best := peak.Load()
			if current <= best || peak.CompareAndSwap(best, current) {
				break
			}
		}
		// Hold the slot so overlapping evaluations are actually observable.
		time.Sleep(hold)
		inFlight.Add(-1)
		return 1, nil
	}
	return eval, &peak
}

func TestEvaluate_ScoresEveryGenome(t *testing.T) {
	pop := testPopulation(t, 50)

	var calls atomic.Int64
	err := neat.Evaluate(context.Background(), pop, func(context.Context, *network.Network) (float64, error) {
		return float64(calls.Add(1)), nil
	})
	require.NoError(t, err)

	assert.Equal(t, int64(50), calls.Load())
	total := 0.0
	for _, fitness := range pop.GenomeFitness {
		assert.NotZero(t, fitness, "every genome must be scored")
		total += fitness
	}
	// 1 + 2 + ... + 50, so every score landed on a distinct genome.
	assert.Equal(t, 1275.0, total)
}

// The whole point of the redesign: genomes really are evaluated in parallel.
func TestEvaluate_RunsConcurrently(t *testing.T) {
	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("needs more than one processor")
	}
	pop := testPopulation(t, 64)

	eval, peak := peakConcurrency(5 * time.Millisecond)
	require.NoError(t, neat.Evaluate(context.Background(), pop, eval))

	assert.Greater(t, peak.Load(), int64(1), "evaluations should overlap")
	assert.LessOrEqual(t, peak.Load(), int64(runtime.GOMAXPROCS(0)),
		"the default should not exceed one worker per processor")
}

// Unlimited means every genome gets its own goroutine, which is what an
// evaluator that blocks on something external wants.
func TestEvaluate_UnlimitedRunsEveryGenomeAtOnce(t *testing.T) {
	pop := testPopulation(t, 40)
	pop.Cfg.Parallelism = neat.Unlimited

	// A barrier that only releases once all 40 evaluations have arrived. If
	// they were not all running at once this would deadlock and time out.
	var (
		arrived  sync.WaitGroup
		released = make(chan struct{})
	)
	arrived.Add(len(pop.Genomes))
	go func() {
		arrived.Wait()
		close(released)
	}()

	done := make(chan error, 1)
	go func() {
		done <- neat.Evaluate(context.Background(), pop, func(ctx context.Context, _ *network.Network) (float64, error) {
			arrived.Done()
			select {
			case <-released:
				return 1, nil
			case <-ctx.Done():
				return 0, ctx.Err()
			}
		})
	}()

	select {
	case err := <-done:
		require.NoError(t, err)
	case <-time.After(10 * time.Second):
		t.Fatal("Unlimited did not run every genome concurrently")
	}
}

func TestEvaluate_RespectsParallelismLimit(t *testing.T) {
	pop := testPopulation(t, 64)
	pop.Cfg.Parallelism = 3

	eval, peak := peakConcurrency(2 * time.Millisecond)
	require.NoError(t, neat.Evaluate(context.Background(), pop, eval))

	assert.LessOrEqual(t, peak.Load(), int64(3))
	assert.Greater(t, peak.Load(), int64(1))
}

func TestEvaluate_PropagatesError(t *testing.T) {
	pop := testPopulation(t, 100)
	boom := errors.New("boom")

	var calls atomic.Int64
	err := neat.Evaluate(context.Background(), pop, func(context.Context, *network.Network) (float64, error) {
		if calls.Add(1) == 10 {
			return 0, boom
		}
		time.Sleep(time.Millisecond)
		return 1, nil
	})

	assert.ErrorIs(t, err, boom)
	// The failure stops the remaining work rather than grinding through it.
	assert.Less(t, calls.Load(), int64(100))
}

func TestEvaluate_StopsOnContextCancellation(t *testing.T) {
	pop := testPopulation(t, 200)

	ctx, cancel := context.WithCancel(context.Background())
	var calls atomic.Int64
	err := neat.Evaluate(ctx, pop, func(context.Context, *network.Network) (float64, error) {
		if calls.Add(1) == 5 {
			cancel()
		}
		time.Sleep(time.Millisecond)
		return 1, nil
	})
	defer cancel()

	assert.ErrorIs(t, err, context.Canceled)
	assert.Less(t, calls.Load(), int64(200))
}

// The evaluator is handed the ctx so a long running evaluation can bail out
// partway through rather than only between genomes.
func TestEvaluate_EvaluatorSeesCancellation(t *testing.T) {
	pop := testPopulation(t, 8)
	pop.Cfg.Parallelism = neat.Unlimited

	ctx, cancel := context.WithCancel(context.Background())
	cancelled := make(chan struct{})
	var once sync.Once

	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	err := neat.Evaluate(ctx, pop, func(ctx context.Context, _ *network.Network) (float64, error) {
		select {
		case <-ctx.Done():
			once.Do(func() { close(cancelled) })
			return 0, ctx.Err()
		case <-time.After(5 * time.Second):
			return 1, nil
		}
	})

	assert.ErrorIs(t, err, context.Canceled)
	select {
	case <-cancelled:
	default:
		t.Fatal("evaluator was never told the context was cancelled")
	}
}

func TestEvaluate_RejectsNilEvaluator(t *testing.T) {
	pop := testPopulation(t, 4)
	assert.ErrorContains(t, neat.Evaluate(context.Background(), pop, nil), "must not be nil")
}

func TestWorkers(t *testing.T) {
	procs := runtime.GOMAXPROCS(0)
	tests := []struct {
		name        string
		parallelism int
		population  int
		want        int
	}{
		{"default is one per processor", 0, 1000, procs},
		{"never more workers than genomes", 0, 2, 2},
		{"explicit count", 4, 1000, 4},
		{"unlimited is one per genome", neat.Unlimited, 37, 37},
		{"empty population needs no workers", 0, 0, 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assert.Equal(t, test.want, neat.Workers(test.parallelism, test.population))
		})
	}
}
