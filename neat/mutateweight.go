package neat

// MutateConnectionWeights returns a copy of genome with perturbed connection
// weights.
func (b *Breeder) MutateConnectionWeights(genome Genome) Genome {
	return b.mutateConnectionWeights(CopyGenome(genome))
}

func (b *Breeder) mutateConnectionWeights(genome Genome) Genome {
	cfg, rng := b.cfg, b.rng
	for i, connection := range genome.Connections {
		if !cfg.Chance(rng, cfg.WeightMutationRate) {
			continue
		}
		if cfg.Chance(rng, cfg.WeightReplaceRate) {
			genome.Connections[i].Weight = cfg.RandWeight(rng)
			continue
		}
		// An additive gaussian step. A multiplicative one can never cross zero
		// and can never move a weight that is already zero, which quietly
		// freezes the sign of every weight for the whole run.
		genome.Connections[i].Weight = cfg.PerturbWeight(rng, connection.Weight)
	}
	return genome
}

// MutateToggleEnabled returns a copy of genome with connection enabled flags
// randomly flipped.
func (b *Breeder) MutateToggleEnabled(genome Genome) Genome {
	return b.mutateToggleEnabled(CopyGenome(genome))
}

func (b *Breeder) mutateToggleEnabled(genome Genome) Genome {
	cfg, rng := b.cfg, b.rng
	for i, connection := range genome.Connections {
		if !cfg.Chance(rng, cfg.EnabledMutationRate) {
			continue
		}
		genome.Connections[i].Enabled = !connection.Enabled
	}
	return genome
}
