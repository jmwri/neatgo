package neat

import "github.com/jmwri/neatgo/util"

func MutateConnectionWeights(cfg Config, genome Genome) Genome {
	genome = CopyGenome(genome)
	for i, connection := range genome.Connections {
		seed := cfg.RandFloatProvider(0, 1)
		if seed > cfg.WeightMutationRate {
			continue
		}

		// Generate a completely new weight, or modify it slightly
		seed = cfg.RandFloatProvider(0, 1)
		newWeight := connection.Weight
		if seed <= cfg.WeightReplaceRate {
			previous := newWeight
			for newWeight == previous {
				newWeight = cfg.RandFloatProvider(cfg.MinWeight, cfg.MaxWeight)
			}
		} else {
			weightAdjustment := -1.0
			isPositiveAdjustment := util.FloatBetween(0, 1) < .5
			if isPositiveAdjustment {
				weightAdjustment = 1
			}
			weightAdjustment = weightAdjustment * (newWeight * cfg.WeightMutationPower)
			newWeight += weightAdjustment
			newWeight += util.RandomGaussian()
			if newWeight > cfg.MaxWeight {
				newWeight = cfg.MaxWeight
			} else if newWeight < cfg.MinWeight {
				newWeight = cfg.MinWeight
			}
		}
		genome.Connections[i].Weight = newWeight
	}
	return genome
}
