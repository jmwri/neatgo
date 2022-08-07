package neat

func MutateConnectionWeights(cfg GenomeConfig, genome Genome) Genome {
	genome = CopyGenome(genome)
	seed := cfg.RandFloatProvider(0, 1)
	if seed > cfg.WeightMutationRate {
		return genome
	}
	for i, connection := range genome.connections {
		// Generate a completely new weight, or modify it slightly
		seed := cfg.RandFloatProvider(0, 1)
		newWeight := connection.Weight
		if seed <= cfg.WeightFullMutationRate {
			previous := newWeight
			for newWeight == previous {
				newWeight = cfg.RandFloatProvider(cfg.MinWeight, cfg.MaxWeight)
			}
		} else {
			newWeight += RandomGaussian() / 50
			if newWeight > cfg.MaxWeight {
				newWeight = cfg.MaxWeight
			} else if newWeight < cfg.MinWeight {
				newWeight = cfg.MinWeight
			}
		}
		genome.connections[i].Weight = newWeight
	}
	return genome
}
