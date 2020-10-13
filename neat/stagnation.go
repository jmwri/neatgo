package neat

import (
	"math"
	"neatgo/aggregation"
	"sort"
)

type SpeciesDataItem struct {
	SpeciesID int64
	Species   *Species
}

type SpeciesResponseItem struct {
	SpeciesID  int64
	Species    *Species
	IsStagnant bool
}

func NewStagnation(cfg *Config, speciesFitnessFn aggregation.Fn) *Stagnation {
	return &Stagnation{
		cfg:              cfg,
		speciesFitnessFn: speciesFitnessFn,
	}
}

type Stagnation struct {
	cfg              *Config
	speciesFitnessFn aggregation.Fn
}

func (s *Stagnation) Update(ss *SpeciesSet, generation int) []SpeciesResponseItem {
	speciesData := make([]SpeciesDataItem, 0)
	for speciesID, species := range ss.species {
		var prevFitness float64
		if len(species.fitnessHistory) > 0 {
			prevFitness = aggregation.Max(species.fitnessHistory)
		} else {
			prevFitness = -math.MaxFloat64
		}

		species.fitness = s.speciesFitnessFn(species.Fitnesses())
		species.fitnessHistory = append(species.fitnessHistory, species.fitness)
		species.adjustedFitness = 0
		if prevFitness == 0 || species.fitness > prevFitness {
			species.lastImprovedGen = generation
		}

		speciesData = append(speciesData, SpeciesDataItem{
			SpeciesID: speciesID,
			Species:   species,
		})
	}

	sort.Slice(speciesData, func(i, j int) bool {
		return speciesData[i].Species.fitness < speciesData[j].Species.fitness
	})

	response := make([]SpeciesResponseItem, 0)
	speciesFitnesses := make([]float64, 0)
	numNonStagnant := len(speciesData)

	for i, data := range speciesData {
		speciesID := data.SpeciesID
		species := data.Species

		stagnantTime := generation - species.lastImprovedGen
		isStagnant := false
		if numNonStagnant > s.cfg.SpeciesElitism {
			isStagnant = stagnantTime >= s.cfg.MaxStagnation
		}

		if len(speciesData)-i <= s.cfg.SpeciesElitism {
			isStagnant = false
		}

		if isStagnant {
			numNonStagnant--
		}

		response = append(response, SpeciesResponseItem{
			SpeciesID:  speciesID,
			Species:    species,
			IsStagnant: isStagnant,
		})
		speciesFitnesses = append(speciesFitnesses, species.fitness)
	}
	return response
}
