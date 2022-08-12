package neat

import (
	"github.com/jmwri/neatgo/util"
)

func MutateNodeBiases(cfg Config, genome Genome) Genome {
	genome = CopyGenome(genome)
	for j, layer := range genome.Layers {
		for i, node := range layer {
			seed := cfg.RandFloatProvider(0, 1)
			if seed > cfg.BiasMutationRate {
				continue
			}

			// Generate a completely new bias, or modify it slightly
			seed = cfg.RandFloatProvider(0, 1)
			newBias := node.Bias
			if seed <= cfg.BiasReplaceRate {
				previous := newBias
				for newBias == previous {
					newBias = cfg.RandFloatProvider(cfg.MinBias, cfg.MaxBias)
				}
			} else {
				biasAdjustment := -1.0
				isPositiveAdjustment := util.FloatBetween(0, 1) < .5
				if isPositiveAdjustment {
					biasAdjustment = 1
				}
				biasAdjustment = biasAdjustment * (newBias * cfg.BiasMutationPower)
				newBias += biasAdjustment
				//newBias += util.RandomGaussian()
				if newBias > cfg.MaxBias {
					newBias = cfg.MaxBias
				} else if newBias < cfg.MinBias {
					newBias = cfg.MinBias
				}
			}
			genome.Layers[j][i].Bias = newBias
		}
	}
	return genome
}
