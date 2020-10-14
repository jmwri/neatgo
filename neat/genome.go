package neat

import (
	"fmt"
	"github.com/jmwri/neatgo/net"
	"math"
)

func NewGenome(n net.NeuralNetwork) *Genome {
	nodeGenes, connectionGenes := GenesFromNetwork(n)
	return &Genome{
		n:               n,
		nodeGenes:       nodeGenes,
		connectionGenes: connectionGenes,
	}
}

func NewGenomeFromConfig(id int64, cfg *Config) (*Genome, error) {
	layerDefinitions := make([]net.LayerDefinition, 0)
	// Add input layer
	layerDefinitions = append(layerDefinitions, net.NewLayerDefinition(cfg.NumInputs, cfg.BiasInitMin, cfg.BiasInitMax, cfg.ActivationDefault, cfg.AggregationDefault))
	// Add hidden layers
	for _, numHidden := range cfg.NumHidden {
		layerDefinitions = append(layerDefinitions, net.NewLayerDefinition(numHidden, cfg.BiasInitMin, cfg.BiasInitMax, cfg.ActivationDefault, cfg.AggregationDefault))
	}
	// Add output layer
	layerDefinitions = append(layerDefinitions, net.NewLayerDefinition(cfg.NumOutputs, cfg.BiasInitMin, cfg.BiasInitMax, cfg.ActivationDefault, cfg.AggregationDefault))

	n, err := net.NewFeedForwardFromDefinition(id, layerDefinitions)
	if err != nil {
		return nil, err
	}

	return NewGenome(n), nil
}

func NewGenomeFromCrossover(id int64, parentA *Genome, parentB *Genome) (*Genome, error) {
	// Always set the fittest parent as `a`
	var a, b *Genome
	if parentA.Fitness() > parentB.Fitness() {
		a = parentA
		b = parentB
	} else {
		a = parentB
		b = parentA
	}

	// Inherit connection genes
	childConnections := make([]net.LayerConnections, 0)
	for lk := range a.connectionGenes {
		childLayerConnections := make(net.LayerConnections, 0)

		// Build up a map of connectionID > connection for both parents
		aLayerConnections := make(map[int64]*ConnectionGene)
		for _, aGene := range a.connectionGenes[lk] {
			aLayerConnections[aGene.ID()] = aGene
		}
		bLayerConnections := make(map[int64]*ConnectionGene)
		for _, bGene := range b.connectionGenes[lk] {
			bLayerConnections[bGene.ID()] = bGene
		}

		for ID, aGene := range aLayerConnections {
			bGene, ok := bLayerConnections[ID]
			if !ok {
				// Excess or disjoint gene: copy from the fittest parent.
				childLayerConnections = append(childLayerConnections, aGene.Copy().c)
			} else {
				// Homologous gene: combine genes from both parents.
				newGene, err := NewConnectionGeneFromCrossover(aGene, bGene)
				if err != nil {
					return nil, fmt.Errorf("failed to crossover connection genes: %w", err)
				}
				childLayerConnections = append(childLayerConnections, newGene.c)
			}
		}

		childConnections = append(childConnections, childLayerConnections)
	}

	// Inherit layer genes
	childLayers := make([]net.Layer, 0)
	for lk := range a.nodeGenes {
		childLayer := make(net.Layer, 0)
		aLayerNodes := make(map[int64]*NodeGene)
		for _, aGene := range a.nodeGenes[lk] {
			aLayerNodes[aGene.ID()] = aGene
		}
		bLayerNodes := make(map[int64]*NodeGene)
		for _, bGene := range b.nodeGenes[lk] {
			bLayerNodes[bGene.ID()] = bGene
		}

		for ID, aGene := range aLayerNodes {
			bGene, ok := bLayerNodes[ID]
			if !ok {
				// Excess or disjoint gene: copy from the fittest parent.
				childLayer = append(childLayer, aGene.Copy().n)
			} else {
				// Homologous gene: combine genes from both parents.
				newGene, err := NewNodeGeneFromCrossover(aGene, bGene)
				if err != nil {
					return nil, fmt.Errorf("failed to crossover node genes: %w", err)
				}
				childLayer = append(childLayer, newGene.n)
			}
		}

		childLayers = append(childLayers, childLayer)
	}

	childNet := net.NewFeedForward(id, childLayers, childConnections)
	return NewGenome(childNet), nil
}

type Genome struct {
	n               net.NeuralNetwork
	nodeGenes       [][]*NodeGene
	connectionGenes [][]*ConnectionGene
	fitness         float64
}

func (g *Genome) ID() int64 {
	return g.n.ID()
}

func (g *Genome) Network() net.NeuralNetwork {
	return g.n
}

func (g *Genome) Copy() *Genome {
	cp := g.n.Copy()
	return NewGenome(cp)
}

func (g *Genome) Mutate(cfg *Config) {
	for i := 0; i < len(g.nodeGenes); i++ {
		for ni := 0; ni < len(g.nodeGenes[i]); ni++ {
			g.nodeGenes[i][ni].Mutate(cfg)
		}
	}
	for i := 0; i < len(g.connectionGenes); i++ {
		for ci := 0; ci < len(g.connectionGenes[i]); ci++ {
			g.connectionGenes[i][ci].Mutate(cfg)
		}
	}
}

func (g *Genome) Fitness() float64 {
	return g.fitness
}

func (g *Genome) Activate(inputs []float64) ([]float64, error) {
	return g.n.Activate(inputs)
}

func (g *Genome) Distance(other *Genome, cfg *Config) float64 {
	nodeDistance := 0.0
	if len(g.nodeGenes) > 0 || len(other.nodeGenes) > 0 {
		disjointNodes := 0
		ownKnownNodes := make(map[int64]*NodeGene)
		for _, layer := range g.nodeGenes {
			for _, node := range layer {
				ownKnownNodes[node.ID()] = node
			}
		}
		otherKnownNodes := make(map[int64]*NodeGene)
		for _, layer := range other.nodeGenes {
			for _, node := range layer {
				otherKnownNodes[node.ID()] = node
			}
		}

		for id := range otherKnownNodes {
			if _, ok := ownKnownNodes[id]; !ok {
				disjointNodes += 1
			}
		}

		for k1, n1 := range ownKnownNodes {
			if n2, ok := otherKnownNodes[k1]; !ok {
				disjointNodes += 1
			} else {
				// Homologous genes compute their own distance value.
				nodeDistance += n1.Distance(n2, cfg)
			}
		}

		maxNodes := math.Max(float64(len(ownKnownNodes)), float64(len(otherKnownNodes)))
		nodeDistance = (nodeDistance + (cfg.CompatibilityDisjointCoefficient * float64(disjointNodes))) / maxNodes
	}

	connectionDistance := 0.0
	if len(g.connectionGenes) > 0 || len(other.connectionGenes) > 0 {
		disjointConnections := 0
		ownKnownConnections := make(map[int64]*ConnectionGene)
		for _, layer := range g.connectionGenes {
			for _, conn := range layer {
				ownKnownConnections[conn.ID()] = conn
			}
		}
		otherKnownConnections := make(map[int64]*ConnectionGene)
		for _, layer := range other.connectionGenes {
			for _, conn := range layer {
				ownKnownConnections[conn.ID()] = conn
			}
		}

		for id := range otherKnownConnections {
			if _, ok := ownKnownConnections[id]; !ok {
				disjointConnections += 1
			}
		}

		for k1, c1 := range ownKnownConnections {
			if c2, ok := otherKnownConnections[k1]; !ok {
				disjointConnections += 1
			} else {
				// Homologous genes compute their own distance value.
				connectionDistance += c1.Distance(c2, cfg)
			}
		}

		maxNodes := math.Max(float64(len(ownKnownConnections)), float64(len(otherKnownConnections)))
		connectionDistance = (connectionDistance + (cfg.CompatibilityDisjointCoefficient * float64(disjointConnections))) / maxNodes
	}

	return nodeDistance + connectionDistance
}
