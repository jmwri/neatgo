package neat

import (
	"math"
	"sort"
)

func NewSpecies(representative Genome) Species {
	return Species{
		AvgFitness:     .0,
		BestFitness:    .0,
		Genomes:        make([]int, 0),
		Representative: representative,
		Staleness:      0,
	}
}

type Species struct {
	AvgFitness     float64
	BestFitness    float64
	Genomes        []int
	Representative Genome
	Staleness      int
}

func Speciate(pop Population) Population {
	for i := range pop.Species {
		pop.Species[i].Genomes = make([]int, 0)
	}
	for i, genome := range pop.Genomes {
		foundSpecies := false
		for j, species := range pop.Species {
			if CompatibleWithSpecies(pop, species, genome) {
				pop.Species[j].Genomes = append(pop.Species[j].Genomes, i)
				foundSpecies = true
				break
			}
		}
		if !foundSpecies {
			species := NewSpecies(genome)
			species.Genomes = append(species.Genomes, i)
			pop.Species = append(pop.Species, species)
		}
	}
	for i, species := range pop.Species {
		bestFitness := 0.0
		totalFitness := 0.0
		for _, genome := range species.Genomes {
			genomeFitness := pop.GenomeFitness[genome]
			totalFitness += genomeFitness
			if genomeFitness > bestFitness {
				bestFitness = genomeFitness
			}
		}
		pop.Species[i].AvgFitness = totalFitness / float64(len(species.Genomes))
		pop.Species[i].BestFitness = bestFitness
	}
	return pop
}

func RankSpecies(pop Population) Population {
	for _, species := range pop.Species {
		// Sort genomes in each species in desc order of fitness
		sort.Slice(species.Genomes, func(i, j int) bool {
			a := species.Genomes[i]
			b := species.Genomes[j]
			return pop.GenomeFitness[a] > pop.GenomeFitness[b]
		})
	}
	// Sort pop.Species in desc order of BestFitness
	sort.Slice(pop.Species, func(i, j int) bool {
		return pop.Species[i].BestFitness > pop.Species[j].BestFitness
	})
	return pop
}

func CompatibleWithSpecies(pop Population, species Species, genome Genome) bool {
	excessAndDisjoint := countExcessAndDisjointGenes(genome, species.Representative)
	averageWeightDiff := calculateAverageConnectionWeightDiff(genome, species.Representative)

	var largeGenomeNormaliser = (genome.NumLayers() + genome.NumNodes()) - 20
	if largeGenomeNormaliser < 1 {
		largeGenomeNormaliser = 1
	}

	// Lower means more similar
	compatibility := (pop.Cfg.SpeciesCompatExcessCoeff * float64(excessAndDisjoint) / float64(largeGenomeNormaliser)) +
		(pop.Cfg.SpeciesCompatWeightDiffCoeff * averageWeightDiff)
	return compatibility <= pop.Cfg.SpeciesCompatThreshold
}

func countExcessAndDisjointGenes(a, b Genome) int {
	innovationNumCount := make(map[int]int)

	for _, node := range a.layers.Nodes() {
		innovationNumCount[node.ID]++
	}
	for _, node := range b.layers.Nodes() {
		innovationNumCount[node.ID]++
	}
	for _, connection := range a.connections {
		innovationNumCount[connection.ID]++
	}
	for _, connection := range b.connections {
		innovationNumCount[connection.ID]++
	}

	tot := 0
	for _, count := range innovationNumCount {
		if count < 2 {
			tot++
		}
	}

	return tot
}

func calculateAverageConnectionWeightDiff(a, b Genome) float64 {
	innovationNumCount := make(map[int]int)
	innovationWeights := make(map[int]float64)
	for _, connection := range a.connections {
		innovationNumCount[connection.ID]++
		innovationWeights[connection.ID] = connection.Weight
	}
	for _, connection := range b.connections {
		innovationNumCount[connection.ID]++
		innovationWeights[connection.ID] -= connection.Weight
	}

	tot := .0
	totalWeightDiff := .0
	for i, count := range innovationNumCount {
		if count == 2 {
			tot++
			totalWeightDiff += math.Abs(innovationWeights[i])
		}
	}

	// Avoid divide by zero
	if tot == 0 {
		return 100
	}
	return totalWeightDiff / tot
}
