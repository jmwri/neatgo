package neat

import (
	"fmt"
	"github.com/jmwri/neatgo/aggregation"
	"math"
	"math/rand"
	"sort"
)

func NewReproduction(cfg *Config, stagnation *Stagnation) *Reproduction {
	return &Reproduction{
		cfg:          cfg,
		currentIndex: 1,
		stagnation:   stagnation,
		ancestors:    make(map[int64]*AncestorPair),
	}
}

type AncestorPair struct {
	A int64
	B int64
}

type Reproduction struct {
	cfg          *Config
	currentIndex int64
	stagnation   *Stagnation
	ancestors    map[int64]*AncestorPair
}

func (r *Reproduction) GetNextIndex() int64 {
	i := r.currentIndex
	r.currentIndex++
	return i
}

func (r *Reproduction) CreateNew(numGenomes int) (map[int64]*Genome, error) {
	newGenomes := make(map[int64]*Genome, 0)
	for i := 0; i < numGenomes; i++ {
		newIndex := r.GetNextIndex()
		genome, err := NewGenomeFromConfig(newIndex, r.cfg)
		if err != nil {
			return newGenomes, err
		}
		newGenomes[newIndex] = genome
		// The new genome doesn't have any parents, so set to nil
		r.ancestors[newIndex] = nil
	}
	return newGenomes, nil
}

func (r *Reproduction) ComputeSpawn(adjustedFitnesses []float64, previousSizes []int, populationSize int, minSpeciesSize int) []int {
	totAdjustedFitness := aggregation.Sum(adjustedFitnesses)

	spawnAmounts := make([]float64, 0)
	for i := 0; i < len(adjustedFitnesses); i++ {
		af := adjustedFitnesses[i]
		ps := float64(previousSizes[i])

		var s float64
		if totAdjustedFitness > 0 {
			s = math.Max(float64(minSpeciesSize), af/totAdjustedFitness*float64(populationSize))
		} else {
			s = float64(minSpeciesSize)
		}

		d := (s - ps) * .5
		c := math.Round(d)

		spawn := ps
		if math.Abs(c) > 0 {
			spawn += c
		} else if d > 0 {
			spawn += 1
		} else if d < 0 {
			spawn -= 1
		}

		spawnAmounts = append(spawnAmounts, spawn)
	}

	totalSpawn := aggregation.Sum(spawnAmounts)
	norm := float64(populationSize) / totalSpawn

	spawnAmountsProcessed := make([]int, 0)
	for _, n := range spawnAmounts {
		normalised := math.Round(n * norm)
		max := math.Max(float64(minSpeciesSize), normalised)
		spawnAmountsProcessed = append(spawnAmountsProcessed, int(max))
	}

	return spawnAmountsProcessed
}

func (r *Reproduction) Reproduce(cfg *Config, speciesSet *SpeciesSet, populationSize int, generation int) map[int64]*Genome {
	// Filter out stagnated species
	allFitnesses := make([]float64, 0)
	remainingSpecies := make([]*Species, 0)
	stagnationResponses := r.stagnation.Update(speciesSet, generation)
	for _, stagResponse := range stagnationResponses {
		if stagResponse.IsStagnant {
			// Don't add to remaining species!
		} else {
			allFitnesses = append(allFitnesses, stagResponse.Species.Fitnesses()...)
			remainingSpecies = append(remainingSpecies, stagResponse.Species)
		}
	}

	if len(remainingSpecies) == 0 {
		speciesSet.species = make(map[int64]*Species)
		return nil
	}

	minFitness := aggregation.Min(allFitnesses)
	maxFitness := aggregation.Max(allFitnesses)

	fitnessRange := math.Max(cfg.FitnessMinDivisor, maxFitness-minFitness)
	adjustedFitnesses := make([]float64, 0)
	previousSizes := make([]int, 0)
	for _, species := range remainingSpecies {
		meanSpeciesFitness := aggregation.Mean(species.Fitnesses())
		adjustedFitness := (meanSpeciesFitness - minFitness) / fitnessRange
		adjustedFitnesses = append(adjustedFitnesses, adjustedFitness)
		species.adjustedFitness = adjustedFitness
		previousSizes = append(previousSizes, len(species.members))
	}

	//averageAdjustedFitness := aggregation.Mean(adjustedFitnesses)
	minSpeciesSize := int(math.Max(float64(cfg.MinSpeciesSize), float64(cfg.Elitism)))

	spawnAmounts := r.ComputeSpawn(adjustedFitnesses, previousSizes, populationSize, minSpeciesSize)

	newPopulation := make(map[int64]*Genome)
	speciesSet.species = make(map[int64]*Species)

	for i := 0; i < len(spawnAmounts); i++ {
		spawnAmount := spawnAmounts[i]
		species := remainingSpecies[i]

		// If elitism is enabled, each species always at least gets to retain its elites.
		spawnAmount = int(math.Max(float64(spawnAmount), float64(cfg.Elitism)))

		if spawnAmount <= 0 {
			panic("cant spawn less than 0")
		}
		oldMembers := make([]*Genome, len(species.members))
		copy(oldMembers, species.members)

		species.members = make([]*Genome, 0)
		speciesSet.species[species.id] = species

		// Sort candidates by min dist
		sort.Slice(oldMembers, func(i, j int) bool {
			return oldMembers[i].Fitness() > oldMembers[j].Fitness()
		})

		// Transfer elites to new generation
		if cfg.Elitism > 0 {
			for i, m := range oldMembers[:cfg.Elitism] {
				newPopulation[int64(i)] = m
				spawnAmount -= 1
			}
		}

		if spawnAmount <= 0 {
			continue
		}

		// Only use the survival threshold fraction to use as parents for the next generation.
		reproCutoff := int(math.Ceil(cfg.SurvivalThreshold * float64(len(oldMembers))))
		// Use at least two parents no matter what the threshold fraction result is.
		if reproCutoff < 2 {
			reproCutoff = 2
		}

		oldMembers = oldMembers[:reproCutoff]

		// Randomly choose parents and produce the number of offspring allotted to the species.
		for spawnAmount > 0 {
			spawnAmount--

			newID := r.GetNextIndex()
			parentAIndex := rand.Intn(len(oldMembers))
			parentA := oldMembers[parentAIndex]
			parentBIndex := rand.Intn(len(oldMembers))
			parentB := oldMembers[parentBIndex]

			child, err := NewGenomeFromCrossover(newID, parentA, parentB)
			if err != nil {
				panic(fmt.Sprintf("failed to crossover genome: %s", err))
			}
			child.Mutate(cfg)
			newPopulation[newID] = child
			r.ancestors[newID] = &AncestorPair{
				A: parentA.ID(),
				B: parentB.ID(),
			}
		}
	}

	return newPopulation
}
