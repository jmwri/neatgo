package neat

import (
	"math"
	"testing"

	"github.com/jmwri/neatgo/v2/network"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Species are ranked by their best current member. A species whose one great
// ancestor has long gone must not stay at the top, and so protected from
// being culled as stale, on the strength of a fitness none of its members has.
func TestRankSpecies_UsesCurrentNotHistoricalFitness(t *testing.T) {
	pop := twoSpeciesPopulation(t, 10,
		[]float64{2, 1}, // once reached 100, now the weaker species
		[]float64{5, 4},
	)
	pop.Species[0].BestFitness = 100
	pop.Species[1].BestFitness = 5

	pop = RankSpecies(pop)

	assert.Equal(t, []int{2, 3}, pop.Species[0].Genomes, "the species with the fittest current member ranks first")
	assert.Equal(t, 100.0, pop.Species[1].BestFitness, "the historical best is kept for staleness tracking")
}

// Both parents come from roulette over the survivors, measured from the worst
// genome in the population, so the weaker of two survivors is still a parent
// some of the time. Measured from the worst survivor instead it would never
// be, and a two-member species would only ever clone its best.
func TestSelectParents_WeakerSurvivorStillBreeds(t *testing.T) {
	pop := twoSpeciesPopulation(t, 10,
		[]float64{6, 4}, // the species under test: two survivors
		[]float64{1, 1}, // the population's worst genomes
	)
	rng := NewRand(1)
	floor := pop.worstFitness()
	assert.Equal(t, 1.0, floor)

	picked := map[int]int{}
	for i := 0; i < 2000; i++ {
		a, b := selectParents(pop, rng, pop.Species[0], floor)
		picked[a]++
		picked[b]++
	}
	assert.Zero(t, picked[2]+picked[3], "parents must come from the species itself")
	// Shares are 5:3 above the floor.
	assert.InDelta(t, 5.0/8, float64(picked[0])/4000, .05)
	assert.InDelta(t, 3.0/8, float64(picked[1])/4000, .05)
}

// A species whose survivors are all at the population's worst fitness has no
// roulette to run, and must fall back to picking uniformly rather than always
// returning the same member.
func TestSelectParents_AllAtFloorPicksUniformly(t *testing.T) {
	pop := twoSpeciesPopulation(t, 10,
		[]float64{-3, -3, -3},
		[]float64{0, 0},
	)
	rng := NewRand(2)
	picked := map[int]int{}
	for i := 0; i < 900; i++ {
		a, _ := selectParents(pop, rng, pop.Species[0], pop.worstFitness())
		picked[a]++
	}
	for _, genomeIndex := range pop.Species[0].Genomes {
		assert.Greater(t, picked[genomeIndex], 200, "genome %d was starved", genomeIndex)
	}
}

// +Inf means "cannot be beaten" and must sanitise to the top of the
// population, not the bottom. NaN and -Inf are broken scores and go to the
// bottom.
func TestSanitiseFitness_KeepsPositiveInfinityOnTop(t *testing.T) {
	pop := Population{GenomeFitness: []float64{math.Inf(1), 3, math.NaN(), 7, math.Inf(-1)}}
	pop = sanitiseFitness(pop)
	assert.Equal(t, []float64{7, 3, 3, 7, 3}, pop.GenomeFitness)

	// Nothing finite at all collapses to zero.
	pop = Population{GenomeFitness: []float64{math.Inf(1), math.NaN()}}
	pop = sanitiseFitness(pop)
	assert.Equal(t, []float64{0, 0}, pop.GenomeFitness)
}

// Input and bias nodes have no bias to differ on, so they must not be averaged
// into the bias term: the same bias difference on the one learnable node has
// to measure the same however many inputs the problem has.
func TestCompatibilityDistance_BiasTermIgnoresFixedNodes(t *testing.T) {
	cfg := DefaultConfig(1, 1)
	cfg.SpeciesCompatExcessCoeff = 0
	cfg.SpeciesCompatWeightDiffCoeff = 0
	cfg.SpeciesCompatBiasDiffCoeff = 1

	build := func(inputs int, outputBias float64) Genome {
		layer := make([]network.Node, 0, inputs+1)
		for i := 0; i < inputs; i++ {
			layer = append(layer, network.Node{ID: i + 1, Type: network.Input, ActivationFn: network.NoActivation})
		}
		layer = append(layer, network.Node{ID: 100, Type: network.Bias, ActivationFn: network.NoActivation})
		return NewGenome([][]network.Node{
			layer,
			{{ID: 200, Type: network.Output, Bias: outputBias, ActivationFn: network.Sigmoid}},
		}, nil)
	}

	narrow := CompatibilityDistance(cfg, build(1, 0), build(1, 2))
	wide := CompatibilityDistance(cfg, build(8, 0), build(8, 2))
	require.InDelta(t, 2.0, narrow, 1e-12)
	assert.InDelta(t, narrow, wide, 1e-12, "extra inputs must not dilute the bias difference")
}
