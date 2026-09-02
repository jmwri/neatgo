package neat

import (
	"cmp"
	"math"
	"slices"
	"sort"

	"github.com/jmwri/neatgo/v2/internal/util"
)

func NewSpecies(representative Genome) Species {
	return Species{
		AvgFitness:     .0,
		BestFitness:    math.Inf(-1),
		Genomes:        make([]int, 0),
		Representative: representative,
		Staleness:      0,
	}
}

type Species struct {
	// AvgFitness is the mean raw fitness of the species' members.
	AvgFitness float64
	// BestFitness is the highest raw fitness in the species.
	BestFitness float64
	// AdjustedFitness is the sum of the members' fitness after sharing, which
	// is what offspring are allocated in proportion to. Since sharing divides
	// each member's fitness by the species size, this is the species' mean raw
	// fitness.
	AdjustedFitness float64
	Genomes         []int
	Representative  Genome
	Staleness       int
}

// Speciate assigns every genome in the population to a species.
//
// Each surviving species keeps a representative drawn from its previous
// members; genomes join the first species whose representative they are
// compatible with, and found a new species otherwise.
func Speciate(pop Population) Population {
	rng := pop.Breeder.rng
	species := make([]Species, 0, len(pop.Species))
	for _, existing := range pop.Species {
		if len(existing.Genomes) == 0 {
			// The species has no members to draw a representative from, so it
			// is extinct.
			continue
		}
		// Set species representative to a random member of the generation that
		// has just been evaluated, then clear the membership list.
		existing.Representative = pop.Genomes[util.RandSliceElement(rng, existing.Genomes)]
		existing.Genomes = make([]int, 0)
		species = append(species, existing)
	}

	// Index every genome and every representative once, rather than rebuilding
	// a lookup table inside each of the thousands of pairwise comparisons
	// below.
	genomeIndexes := make([]geneIndex, len(pop.Genomes))
	for i, genome := range pop.Genomes {
		genomeIndexes[i] = newGeneIndex(genome)
	}
	representatives := make([]geneIndex, len(species))
	for i := range species {
		representatives[i] = newGeneIndex(species[i].Representative)
	}

	for i, genome := range pop.Genomes {
		found := false
		for j := range species {
			if compatibility(pop.Cfg, genomeIndexes[i], representatives[j]) <= pop.Cfg.SpeciesCompatThreshold {
				species[j].Genomes = append(species[j].Genomes, i)
				found = true
				break
			}
		}
		if !found {
			newSpecies := NewSpecies(genome)
			newSpecies.Genomes = append(newSpecies.Genomes, i)
			species = append(species, newSpecies)
			representatives = append(representatives, genomeIndexes[i])
		}
	}

	// Drop any carried-over species that attracted no members this generation,
	// then refresh each survivor's fitness statistics and staleness.
	surviving := make([]Species, 0, len(species))
	for _, s := range species {
		if len(s.Genomes) == 0 {
			continue
		}

		bestFitness := math.Inf(-1)
		totalFitness := 0.0
		for _, genomeIndex := range s.Genomes {
			fitness := pop.GenomeFitness[genomeIndex]
			totalFitness += fitness
			if fitness > bestFitness {
				bestFitness = fitness
			}
		}

		// A species is stale while its best member fails to beat the best it
		// has ever produced.
		if bestFitness > s.BestFitness {
			s.BestFitness = bestFitness
			s.Staleness = 0
		} else {
			s.Staleness++
		}
		s.AvgFitness = totalFitness / float64(len(s.Genomes))
		surviving = append(surviving, s)
	}

	pop.Species = surviving
	return pop
}

// AdjustCompatThreshold nudges the compatibility threshold towards whatever
// keeps the number of species near cfg.TargetSpecies.
//
// A fixed threshold is fragile: the same value that yields a healthy handful of
// species at the start will shatter the population into dozens once genomes
// have grown, and once species are down to two or three members each, almost
// every slot goes to elites and the search stops making progress. Retargeting
// the threshold each generation keeps speciation doing its actual job of
// protecting innovation rather than fragmenting the population.
func AdjustCompatThreshold(pop Population) Population {
	if pop.Cfg.TargetSpecies <= 0 || pop.Cfg.SpeciesCompatThresholdAdjust <= 0 {
		return pop
	}
	switch {
	case len(pop.Species) > pop.Cfg.TargetSpecies:
		pop.Cfg.SpeciesCompatThreshold += pop.Cfg.SpeciesCompatThresholdAdjust
	case len(pop.Species) < pop.Cfg.TargetSpecies:
		pop.Cfg.SpeciesCompatThreshold -= pop.Cfg.SpeciesCompatThresholdAdjust
	}
	if pop.Cfg.SpeciesCompatThreshold < pop.Cfg.MinSpeciesCompatThreshold {
		pop.Cfg.SpeciesCompatThreshold = pop.Cfg.MinSpeciesCompatThreshold
	}
	return pop
}

// RankSpecies sorts the genomes within each species, and the species
// themselves, from fittest to least fit.
//
// Species are ranked by the fitness of their best current member, which is
// what the rank is used for: deciding which species are doing well enough now
// to be protected from being culled as stale, and which breed first when the
// population is topped up. Ranking by the best a species has ever produced
// would keep a species whose members have since scattered to other species,
// or been overtaken, at the top for the rest of the run on the strength of
// one ancestor.
//
// Both sorts are stable, so equally fit genomes keep their index order and
// equally fit species their existing order. Which of two tied genomes is the
// elite decides what the next generation is built from, so it must follow from
// the population and not from which sort algorithm happened to be in use.
func RankSpecies(pop Population) Population {
	for i := range pop.Species {
		slices.SortStableFunc(pop.Species[i].Genomes, func(a, b int) int {
			return cmp.Compare(pop.GenomeFitness[b], pop.GenomeFitness[a])
		})
	}
	currentBest := func(species Species) float64 {
		if len(species.Genomes) == 0 {
			return math.Inf(-1)
		}
		return pop.GenomeFitness[species.Genomes[0]]
	}
	slices.SortStableFunc(pop.Species, func(a, b Species) int {
		return cmp.Compare(currentBest(b), currentBest(a))
	})
	return pop
}

// FitnessSharing computes each genome's adjusted fitness: its raw fitness
// divided by the size of its species.
//
// Sharing is what stops a single successful topology from swamping the
// population. Offspring go to a species in proportion to the sum of its
// members' adjusted fitness, which is its mean raw fitness: a species twice
// the size earns no more for it, and has to be fitter per member to grow. The
// sum is what the paper allocates by. Dividing it by the size again, to get
// a mean of the adjusted values, penalises size twice over: a species then
// loses offspring for having grown, shrinks, wins them back, and the
// population oscillates between species instead of settling where the
// fitness says it should. The raw fitness is left untouched so that reporting
// and best-genome tracking stay honest.
func FitnessSharing(pop Population) Population {
	for i := range pop.GenomeAdjustedFitness {
		pop.GenomeAdjustedFitness[i] = 0
	}
	for i, species := range pop.Species {
		if len(species.Genomes) == 0 {
			continue
		}
		size := float64(len(species.Genomes))
		sum := 0.0
		for _, genomeIndex := range species.Genomes {
			adjusted := pop.GenomeFitness[genomeIndex] / size
			pop.GenomeAdjustedFitness[genomeIndex] = adjusted
			sum += adjusted
		}
		pop.Species[i].AdjustedFitness = sum
	}
	return pop
}

// CullSpecies removes the least fit members of each species so that only the
// top SurvivalThreshold fraction may reproduce. At least two members are kept
// where possible so that crossover still has two parents to work with.
func CullSpecies(pop Population) Population {
	for i, species := range pop.Species {
		if len(species.Genomes) == 0 {
			continue
		}
		keep := int(math.Ceil(pop.Cfg.SurvivalThreshold * float64(len(species.Genomes))))
		if keep < 2 {
			keep = 2
		}
		if keep > len(species.Genomes) {
			keep = len(species.Genomes)
		}
		pop.Species[i].Genomes = species.Genomes[:keep]
	}
	return pop
}

// KillStaleSpecies removes species that have not improved for
// SpeciesStalenessThreshold generations, always keeping the SpeciesElitism
// fittest species alive. If every species is stale the elitism floor is what
// stops the population from going extinct.
func KillStaleSpecies(pop Population) Population {
	keep := make([]Species, 0, len(pop.Species))
	for i, species := range pop.Species {
		if i < pop.Cfg.SpeciesElitism || species.Staleness < pop.Cfg.SpeciesStalenessThreshold {
			keep = append(keep, species)
		}
	}
	if len(keep) == 0 {
		return pop
	}
	pop.Species = keep
	return pop
}

// getDesiredOffspringCount allocates the whole population across the species in
// proportion to their adjusted fitness.
//
// The allocation is exact: largest-remainder rounding distributes the leftover
// slots, so the population size never drifts. Fitnesses are shifted so that the
// worst genome in the population sits at zero, because a fitness function that
// returns negative values would otherwise produce negative or nonsensical
// shares. Shifting by the worst genome rather than the worst species means a
// species only gets nothing when every member of it is as bad as the
// population's worst; shifting by the worst species would hand the second
// best of two species the minimum allowance no matter how close it was.
func getDesiredOffspringCount(pop Population) []int {
	counts := make([]int, len(pop.Species))
	if len(pop.Species) == 0 {
		return counts
	}

	minFitness := math.Inf(1)
	for _, fitness := range pop.GenomeFitness {
		if fitness < minFitness {
			minFitness = fitness
		}
	}
	for _, species := range pop.Species {
		// A species' adjusted fitness is its mean, which can only be below
		// the population's minimum if the population has been replaced since
		// it was computed; keep the shares non-negative either way.
		if species.AdjustedFitness < minFitness {
			minFitness = species.AdjustedFitness
		}
	}

	// Every species gets a small floor share so that a species which is merely
	// the worst is not instantly wiped out.
	const epsilon = 1e-9
	shares := make([]float64, len(pop.Species))
	total := 0.0
	for i, species := range pop.Species {
		shares[i] = species.AdjustedFitness - minFitness + epsilon
		total += shares[i]
	}

	// Distribute whole slots first, tracking the fractional remainder.
	type remainder struct {
		index int
		frac  float64
	}
	remainders := make([]remainder, len(pop.Species))
	allocated := 0
	for i := range pop.Species {
		exact := shares[i] / total * float64(pop.Cfg.PopulationSize)
		counts[i] = int(math.Floor(exact))
		allocated += counts[i]
		remainders[i] = remainder{index: i, frac: exact - math.Floor(exact)}
	}

	// Then hand out what is left over, largest remainder first.
	sort.SliceStable(remainders, func(a, b int) bool {
		return remainders[a].frac > remainders[b].frac
	})
	for i := 0; allocated < pop.Cfg.PopulationSize; i = (i + 1) % len(remainders) {
		counts[remainders[i].index]++
		allocated++
	}

	// Apply the minimum species size, then claw the extra slots back from the
	// largest allocations so the total still matches the population size.
	for i := range counts {
		if counts[i] < pop.Cfg.MinSpeciesSize {
			allocated += pop.Cfg.MinSpeciesSize - counts[i]
			counts[i] = pop.Cfg.MinSpeciesSize
		}
	}
	for allocated > pop.Cfg.PopulationSize {
		largest := 0
		for i := range counts {
			if counts[i] > counts[largest] {
				largest = i
			}
		}
		if counts[largest] <= 1 {
			// Nothing left to take without emptying a species.
			break
		}
		counts[largest]--
		allocated--
	}

	return counts
}

// GetOffspring produces a single mutated child from the given species.
func GetOffspring(pop Population, species Species) Genome {
	breeder := pop.Breeder
	return breeder.mutateStructure(breeder.breed(pop, species, pop.worstFitness()))
}

// breed selects parents, produces a child, and applies every mutation that does
// not change the genome's structure.
//
// This is the expensive half of making an offspring and it touches no shared
// state beyond the read-only population, so many can run at once. The
// structural half is applied separately by mutateStructure.
//
// floor is the population's worst fitness, which parent selection measures
// fitness from; it is passed in so a generation computes it once.
func (b *Breeder) breed(pop Population, species Species, floor float64) Genome {
	rng := b.rng
	if len(species.Genomes) == 0 {
		return Genome{}
	}

	var child Genome
	if len(species.Genomes) > 1 && b.cfg.Chance(rng, b.cfg.MateCrossoverRate) {
		fitter, other := selectParents(pop, rng, species, floor)
		if pop.GenomeFitness[fitter] < pop.GenomeFitness[other] {
			fitter, other = other, fitter
		}
		// Crossover builds a brand new genome, so it is already ours to mutate.
		child = b.Crossover(pop.Genomes[fitter], pop.Genomes[other])
	} else {
		// A straight clone still shares its slices with the parent, so it has
		// to be copied before being mutated.
		child = CopyGenome(pop.Genomes[util.RandSliceElement(rng, species.Genomes)])
	}

	return b.mutateParameters(child)
}

// selectParents picks two parents from the species' surviving members by
// roulette over their fitness.
//
// Fitness is measured from the worst genome in the whole population, the same
// floor the offspring allocation uses, so that every survivor has a share
// unless it is as bad as the population's worst. Measuring from the worst
// survivor instead, as this once did, gives that survivor no share at all -
// and a species cut down to two members, the most common size late in a run,
// then only ever mates its best with itself. Its crossover is a clone, and the
// second parent the cull kept for it is never used.
//
// Picking uniformly among survivors, as some implementations do, was tried
// and is measurably slower to find a solution: the cull alone applies too
// little pressure within a species.
func selectParents(pop Population, rng *Rand, species Species, floor float64) (int, int) {
	return rouletteParent(pop, rng, species, floor), rouletteParent(pop, rng, species, floor)
}

func rouletteParent(pop Population, rng *Rand, species Species, floor float64) int {
	total := 0.0
	for _, genomeIndex := range species.Genomes {
		total += pop.GenomeFitness[genomeIndex] - floor
	}
	if total <= 0 {
		// Every member is as fit as the worst in the population, so pick
		// uniformly.
		return util.RandSliceElement(rng, species.Genomes)
	}
	chosen := util.FloatBetween(rng, 0, total)
	running := 0.0
	for _, genomeIndex := range species.Genomes {
		running += pop.GenomeFitness[genomeIndex] - floor
		if running > chosen {
			return genomeIndex
		}
	}
	return species.Genomes[len(species.Genomes)-1]
}
