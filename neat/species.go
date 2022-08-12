package neat

import (
	"github.com/jmwri/neatgo/util"
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
		oldBestFitness := species.BestFitness
		oldAvgFitness := species.AvgFitness
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

		// If the species didn't get a new max, or increased average, then mark it as stale.
		bestImproved := pop.Species[i].BestFitness > oldBestFitness
		avgImproved := pop.Species[i].AvgFitness > oldAvgFitness
		improved := bestImproved || avgImproved
		if !improved {
			pop.Species[i].Staleness++
		} else {
			pop.Species[i].Staleness = 0
		}
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

func CullSpecies(pop Population) Population {
	for i, species := range pop.Species {
		if len(species.Genomes) <= 2 {
			continue
		}
		removeFromIndex := len(species.Genomes) / 2
		pop.Species[i].Genomes = pop.Species[i].Genomes[removeFromIndex:]
	}
	return pop
}

func FitnessSharing(pop Population) Population {
	for i, species := range pop.Species {
		avgFitness := 0.0
		for _, genomeIndex := range species.Genomes {
			pop.GenomeFitness[genomeIndex] = pop.GenomeFitness[genomeIndex] / float64(len(species.Genomes))
			avgFitness += pop.GenomeFitness[genomeIndex]
		}
		pop.Species[i].AvgFitness = avgFitness / float64(len(species.Genomes))
	}
	return pop
}

func KillStaleSpecies(pop Population) Population {
	keepSpecies := make([]Species, 0)
	removedSpecies := make([]Species, 0)
	for _, species := range pop.Species {
		if species.Staleness < pop.Cfg.SpeciesStalenessThreshold {
			keepSpecies = append(keepSpecies, species)
		} else {
			removedSpecies = append(removedSpecies, species)
		}
	}

	// If we're killing too many species, add some stale ones back in.
	for i := len(keepSpecies); i < pop.Cfg.MinSpecies; i++ {
		if i >= len(removedSpecies) {
			break
		}
		keepSpecies = append(keepSpecies, removedSpecies[i])
	}

	pop.Species = keepSpecies

	return pop
}

func KillBadSpecies(pop Population) Population {
	desiredOffspring := getDesiredOffspringCount(pop)
	keepSpecies := make([]Species, 0)
	for i, species := range pop.Species {
		if i < pop.Cfg.MinSpecies {
			// Always leave at least the required species alive
			keepSpecies = append(keepSpecies, species)
			continue
		}
		numOffspring, ok := desiredOffspring[i]
		if !ok {
			continue
		}
		if numOffspring < 1 {
			continue
		}
		keepSpecies = append(keepSpecies, species)
	}

	pop.Species = keepSpecies

	return pop
}

func getDesiredOffspringCount(pop Population) map[int]int {
	avgFitnessSum := 0.0
	for _, species := range pop.Species {
		avgFitnessSum += species.AvgFitness
	}

	desiredOffspring := make(map[int]int)
	for i, species := range pop.Species {
		offspringCount := int(math.Floor(species.AvgFitness / avgFitnessSum * float64(pop.Cfg.PopulationSize)))
		desiredOffspring[i] = offspringCount
	}
	return desiredOffspring
}

func CompatibleWithSpecies(pop Population, species Species, genome Genome) bool {
	excessAndDisjoint := countExcessAndDisjointGenes(genome, species.Representative)
	averageWeightDiff := calculateAverageConnectionWeightDiff(genome, species.Representative)
	averageBiasDiff := calculateAverageNodeBiasDiff(genome, species.Representative)

	var largeGenomeNormaliser = (genome.NumLayers() + genome.NumNodes()) - 20
	if largeGenomeNormaliser < 1 {
		largeGenomeNormaliser = 1
	}

	excessAndDisjointDiff := pop.Cfg.SpeciesCompatExcessCoeff * float64(excessAndDisjoint) / float64(largeGenomeNormaliser)
	weightDiff := pop.Cfg.SpeciesCompatWeightDiffCoeff * averageWeightDiff
	biasDiff := pop.Cfg.SpeciesCompatBiasDiffCoeff * averageBiasDiff

	// Lower means more similar
	compatibility := excessAndDisjointDiff + weightDiff + biasDiff
	return compatibility <= pop.Cfg.SpeciesCompatThreshold
}

func countExcessAndDisjointGenes(a, b Genome) int {
	innovationNumCount := make(map[int]int)

	for _, node := range a.Layers.Nodes() {
		innovationNumCount[node.ID]++
	}
	for _, node := range b.Layers.Nodes() {
		innovationNumCount[node.ID]++
	}
	for _, connection := range a.Connections {
		innovationNumCount[connection.ID]++
	}
	for _, connection := range b.Connections {
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
	for _, connection := range a.Connections {
		innovationNumCount[connection.ID]++
		innovationWeights[connection.ID] = connection.Weight
	}
	for _, connection := range b.Connections {
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

func calculateAverageNodeBiasDiff(a, b Genome) float64 {
	innovationNumCount := make(map[int]int)
	innovationBiases := make(map[int]float64)
	for _, layer := range a.Layers {
		for _, node := range layer {
			innovationNumCount[node.ID]++
			innovationBiases[node.ID] = node.Bias
		}
	}
	for _, layer := range b.Layers {
		for _, node := range layer {
			innovationNumCount[node.ID]++
			innovationBiases[node.ID] -= node.Bias
		}
	}

	tot := .0
	totalBiasDiff := .0
	for i, count := range innovationNumCount {
		if count == 2 {
			tot++
			totalBiasDiff += math.Abs(innovationBiases[i])
		}
	}

	// Avoid divide by zero
	if tot == 0 {
		return 100
	}
	return totalBiasDiff / tot
}

func GetOffspring(pop Population, species Species) Genome {
	performCrossover := util.FloatBetween(0, 1) < pop.Cfg.MateCrossoverRate
	var baby Genome
	if performCrossover {
		a := getRandomSpeciesOffspring(pop, species)
		b := getRandomSpeciesOffspring(pop, species)
		if pop.GenomeFitness[a] < pop.GenomeFitness[b] {
			a, b = b, a
		}
		aGenome := pop.Genomes[a]
		bGenome := pop.Genomes[b]
		baby = Crossover(pop.Cfg, aGenome, bGenome)
	} else {
		randomGenome := getRandomSpeciesOffspring(pop, species)
		baby = CopyGenome(pop.Genomes[randomGenome])
	}
	return MutateGenome(pop.Cfg, baby)
}

func getRandomSpeciesOffspring(pop Population, species Species) int {
	fitnessSum := 0.0
	for _, genomeID := range species.Genomes {
		fitnessSum += pop.GenomeFitness[genomeID]
	}
	chosenFitness := util.FloatBetween(0, fitnessSum)
	pickSum := 0.0
	for _, genomeID := range species.Genomes {
		pickSum += pop.GenomeFitness[genomeID]
		if pickSum > chosenFitness {
			return genomeID
		}
	}
	return species.Genomes[0]
}
