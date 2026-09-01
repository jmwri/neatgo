package neat_test

import (
	"testing"

	"github.com/jmwri/neatgo/v2/neat"
	"github.com/stretchr/testify/assert"
)

// The whole point of a historical marking is that the same structural change
// gets the same ID whoever discovers it. Without this, crossover cannot align
// two genomes and the compatibility distance is meaningless.
func TestInnovations_SameStructureGetsSameID(t *testing.T) {
	innovations := neat.NewInnovations(neat.NewSequentialIDProvider())

	first := innovations.ConnectionID(1, 2)
	second := innovations.ConnectionID(1, 2)
	assert.Equal(t, first, second, "the same connection must reuse its innovation number")

	other := innovations.ConnectionID(1, 3)
	assert.NotEqual(t, first, other, "a different connection must get its own innovation number")

	assert.Equal(t, innovations.ConnectionID(2, 1), innovations.ConnectionID(2, 1))
	assert.NotEqual(t, innovations.ConnectionID(1, 2), innovations.ConnectionID(2, 1), "direction matters")
}

func TestInnovations_SameSplitGetsSameGenes(t *testing.T) {
	innovations := neat.NewInnovations(neat.NewSequentialIDProvider())

	connectionID := innovations.ConnectionID(1, 2)
	first := innovations.SplitConnection(connectionID, 1, 2)
	second := innovations.SplitConnection(connectionID, 1, 2)
	assert.Equal(t, first, second, "splitting the same connection twice must yield the same three genes")

	assert.NotEqual(t, first.NodeID, first.InConnection)
	assert.NotEqual(t, first.InConnection, first.OutConnection)

	// The halves of the split are themselves connections, so a later add
	// connection mutation producing the same edge must reuse their IDs.
	assert.Equal(t, first.InConnection, innovations.ConnectionID(1, first.NodeID))
	assert.Equal(t, first.OutConnection, innovations.ConnectionID(first.NodeID, 2))
}

// A population generated genome-by-genome would share no gene IDs at all, so
// every genome would look maximally different from every other and the
// population would shatter into one species per genome before evolution began.
func TestGeneratePopulation_SharesOneGenotype(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.PopulationSize = 50

	pop, err := neat.GeneratePopulation(cfg)
	assert.NoError(t, err)
	assert.Len(t, pop.Genomes, 50)

	first := pop.Genomes[0]
	firstNodeIDs := nodeIDs(first)
	firstConnectionIDs := connectionIDs(first)

	identicalWeights := 0
	for _, genome := range pop.Genomes {
		assert.Equal(t, firstNodeIDs, nodeIDs(genome), "every genome must share the same node genes")
		assert.Equal(t, firstConnectionIDs, connectionIDs(genome), "every genome must share the same connection genes")
		for i, connection := range genome.Connections {
			if connection.Weight == first.Connections[i].Weight {
				identicalWeights++
			}
		}
	}
	// Structure is shared, but the weights are not.
	assert.Less(t, identicalWeights, len(first.Connections)*3, "weights should be drawn independently per genome")

	// Which means the initial population is a single species.
	pop = neat.Speciate(pop)
	assert.Len(t, pop.Species, 1, "an initial population differing only in weights is one species")
}

func nodeIDs(genome neat.Genome) []int {
	ids := make([]int, 0, genome.NumNodes())
	for _, node := range genome.Layers.Nodes() {
		ids = append(ids, node.ID)
	}
	return ids
}

func connectionIDs(genome neat.Genome) []int {
	ids := make([]int, 0, len(genome.Connections))
	for _, connection := range genome.Connections {
		ids = append(ids, connection.ID)
	}
	return ids
}

// A genome must never hold two copies of the same gene. Because historical
// markings are stable, splitting a connection that was re-enabled after an
// earlier split would otherwise hand back the node ID the genome already has.
func TestMutateAddNode_NeverDuplicatesAGene(t *testing.T) {
	cfg := neat.DefaultConfig(2, 1)
	cfg.BiasNodes = 0
	cfg.AddNodeMutationRate = 1
	// Aggressively flip enabled flags so re-enabled splits are exercised.
	cfg.EnabledMutationRate = .5
	cfg.DeleteNodeMutationRate = 0
	cfg.DeleteConnectionMutationRate = 0
	cfg.AddConnectionMutationRate = .5
	breeder := neat.NewBreeder(cfg, neat.NewRand(1), nil)

	genome, err := breeder.NewGenome()
	assert.NoError(t, err)

	for i := 0; i < 300; i++ {
		genome = breeder.MutateGenome(genome)
		assertNoDuplicateGenes(t, genome, i)
	}
}

func assertNoDuplicateGenes(t *testing.T, genome neat.Genome, iteration int) {
	t.Helper()
	seenNodes := make(map[int]bool)
	for _, node := range genome.Layers.Nodes() {
		assert.Falsef(t, seenNodes[node.ID], "duplicate node gene %d at iteration %d", node.ID, iteration)
		seenNodes[node.ID] = true
	}
	seenConnections := make(map[int]bool)
	for _, connection := range genome.Connections {
		assert.Falsef(t, seenConnections[connection.ID], "duplicate connection gene %d at iteration %d", connection.ID, iteration)
		seenConnections[connection.ID] = true
		assert.Truef(t, seenNodes[connection.From], "connection %d references missing node %d at iteration %d", connection.ID, connection.From, iteration)
		assert.Truef(t, seenNodes[connection.To], "connection %d references missing node %d at iteration %d", connection.ID, connection.To, iteration)
	}
}
