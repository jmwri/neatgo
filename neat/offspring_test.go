package neat

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// twoSpeciesPopulation builds a population of two species with the given
// member fitnesses, ready for offspring allocation.
func twoSpeciesPopulation(t *testing.T, size int, a, b []float64) Population {
	t.Helper()
	cfg := DefaultConfig(2, 1)
	cfg.PopulationSize = size
	cfg.MinSpeciesSize = 1
	pop := Population{
		Cfg:                   cfg,
		GenomeFitness:         append(append([]float64{}, a...), b...),
		GenomeAdjustedFitness: make([]float64, len(a)+len(b)),
	}
	first, second := NewSpecies(Genome{}), NewSpecies(Genome{})
	for i := range a {
		first.Genomes = append(first.Genomes, i)
	}
	for i := range b {
		second.Genomes = append(second.Genomes, len(a)+i)
	}
	pop.Species = []Species{first, second}
	require.NotEmpty(t, pop.Species)
	return pop
}

// Offspring are allocated in proportion to the sum of adjusted fitness, which
// is the species' mean. A species that is twice the size but no fitter per
// member must earn the same allowance, not half of it: dividing by the size a
// second time would make every large species shrink and every small one grow,
// regardless of fitness.
func TestOffspring_SizeAloneDoesNotChangeTheShare(t *testing.T) {
	pop := twoSpeciesPopulation(t, 100,
		[]float64{10, 10, 10, 10, 10, 10, 10, 10},
		[]float64{10, 10},
	)
	pop = FitnessSharing(pop)
	assert.InDelta(t, pop.Species[0].AdjustedFitness, pop.Species[1].AdjustedFitness, 1e-12,
		"equal mean fitness must give equal adjusted fitness whatever the size")

	counts := getDesiredOffspringCount(pop)
	assert.Equal(t, 50, counts[0])
	assert.Equal(t, 50, counts[1])
}

// A fitter species earns proportionally more, measured from the worst genome
// in the population rather than from the worst species, so that the runner-up
// of two species is not automatically cut to the minimum.
func TestOffspring_ShareFollowsMeanFitnessAboveTheWorstGenome(t *testing.T) {
	pop := twoSpeciesPopulation(t, 100,
		[]float64{4, 4}, // mean 4, three above the worst genome
		[]float64{1, 3}, // mean 2, one above the worst genome
	)
	pop = FitnessSharing(pop)
	counts := getDesiredOffspringCount(pop)
	assert.Equal(t, 75, counts[0])
	assert.Equal(t, 25, counts[1])
	assert.Equal(t, 100, counts[0]+counts[1])
}

// Negative fitness must still produce a valid, exact allocation.
func TestOffspring_NegativeFitnessAllocatesExactly(t *testing.T) {
	pop := twoSpeciesPopulation(t, 30,
		[]float64{-5, -1},
		[]float64{-9, -9},
	)
	pop = FitnessSharing(pop)
	counts := getDesiredOffspringCount(pop)
	total := 0
	for _, count := range counts {
		assert.GreaterOrEqual(t, count, pop.Cfg.MinSpeciesSize)
		total += count
	}
	assert.Equal(t, 30, total)
	// The species entirely made of the worst genomes gets only the floor.
	assert.Equal(t, pop.Cfg.MinSpeciesSize, counts[1])
}

// Trimming an over-full generation must keep species membership and the genome
// list in step: the last species owns the tail of the genome list, and a
// species emptied by the trim has to go with its genome.
func TestEvolve_TrimsOverfullGenerationConsistently(t *testing.T) {
	cfg := DefaultConfig(2, 1)
	cfg.PopulationSize = 6
	cfg.MinSpeciesSize = 1
	cfg.Elitism = 0
	pop, err := GeneratePopulation(cfg)
	require.NoError(t, err)

	// Six singleton species, then the caller shrinks the population to three
	// between generations. Every species is owed at least one slot and the
	// allocation cannot cut a species below one member, so Evolve is handed
	// more genomes than fit and has to trim.
	pop.Cfg.PopulationSize = 3
	pop.Species = nil
	for i := range pop.Genomes {
		species := NewSpecies(pop.Genomes[i])
		species.Genomes = []int{i}
		species.BestFitness = 1
		species.AdjustedFitness = 1
		pop.Species = append(pop.Species, species)
	}
	for i := range pop.GenomeFitness {
		pop.GenomeFitness[i] = 1
	}
	pop.BestEverGenomeFitness = math.Inf(-1)

	pop = Evolve(pop)

	assert.Len(t, pop.Genomes, 3)
	seen := make(map[int]bool)
	for i, species := range pop.Species {
		assert.NotEmpty(t, species.Genomes, "species %d was left with no members", i)
		for _, genomeIndex := range species.Genomes {
			assert.Less(t, genomeIndex, len(pop.Genomes), "species %d refers past the end of the population", i)
			assert.False(t, seen[genomeIndex], "genome %d is claimed by two species", genomeIndex)
			seen[genomeIndex] = true
		}
	}
	for i, genome := range pop.Genomes {
		assert.NotZero(t, genome.NumLayers(), "genome %d was never filled in", i)
	}
}
