package neat

import (
	"github.com/jmwri/neatgo/v2/network"
)

// MutateAddConnection returns a copy of genome with a new connection added.
func (b *Breeder) MutateAddConnection(genome Genome) Genome {
	return b.mutateAddConnection(CopyGenome(genome))
}

func (b *Breeder) mutateAddConnection(genome Genome) Genome {
	cfg, rng := b.cfg, b.rng
	if !cfg.Chance(rng, cfg.AddConnectionMutationRate) {
		return genome
	}

	connectionToAdd, ok := pickPotentialConnection(rng, genome)
	// No potential connection, so no mutation.
	if !ok {
		return genome
	}
	connection := network.NewConnection(
		b.innovations.ConnectionID(connectionToAdd.from, connectionToAdd.to),
		connectionToAdd.from,
		connectionToAdd.to,
		cfg.RandWeight(rng),
		true,
	)

	genome.Connections = append(genome.Connections, connection)
	return genome
}

type potentialConnection struct {
	from, to int
}

// pickPotentialConnection chooses uniformly among the connections that could be
// added without creating a cycle: any node in an earlier layer to any node in a
// later one.
//
// Restricting this to adjacent layers only would forbid skip connections
// entirely, cutting out a large and useful part of the topology space -
// including the minimal XOR solution, where an input feeds the output directly
// as well as through a hidden node.
//
// The candidates are counted and then walked a second time, rather than
// collected into a slice. There are up to O(nodes^2) of them and exactly one is
// wanted, so building the list allocates a large slice per mutation to throw
// nearly all of it away.
func pickPotentialConnection(rng *Rand, genome Genome) (potentialConnection, bool) {
	existing := make(map[connectionKey]struct{}, len(genome.Connections))
	for _, connection := range genome.Connections {
		existing[connectionKey{from: connection.From, to: connection.To}] = struct{}{}
	}

	count := 0
	forEachPotentialConnection(genome, existing, func(potentialConnection) bool {
		count++
		return true
	})
	if count == 0 {
		return potentialConnection{}, false
	}

	wanted := rng.IntN(count)
	var chosen potentialConnection
	forEachPotentialConnection(genome, existing, func(candidate potentialConnection) bool {
		if wanted == 0 {
			chosen = candidate
			return false
		}
		wanted--
		return true
	})
	return chosen, true
}

// forEachPotentialConnection visits every addable connection until visit
// returns false.
func forEachPotentialConnection(genome Genome, existing map[connectionKey]struct{}, visit func(potentialConnection) bool) {
	for toLayer := 1; toLayer < len(genome.Layers); toLayer++ {
		for fromLayer := 0; fromLayer < toLayer; fromLayer++ {
			for _, fromNode := range genome.Layers[fromLayer] {
				for _, toNode := range genome.Layers[toLayer] {
					if _, ok := existing[connectionKey{from: fromNode.ID, to: toNode.ID}]; ok {
						continue
					}
					if !visit(potentialConnection{from: fromNode.ID, to: toNode.ID}) {
						return
					}
				}
			}
		}
	}
}
