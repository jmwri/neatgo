package neat

import (
	"fmt"
	"github.com/jmwri/neatgo/activation"
	"github.com/jmwri/neatgo/aggregation"
	"github.com/jmwri/neatgo/net"
	"github.com/jmwri/neatgo/util"
	"math"
)

func NewNodeGene(n *net.Node) *NodeGene {
	return &NodeGene{
		n: n,
	}
}

func NewNodeGeneFromCrossover(parentA *NodeGene, parentB *NodeGene) (*NodeGene, error) {
	if parentA.ID() != parentB.ID() {
		return nil, fmt.Errorf("cant crossover gene if IDs do not match")
	}

	newBias := parentA.n.Bias()
	if util.RandFloat(0, 1) < 0.5 {
		newBias = parentB.n.Bias()
	}
	newActivationFn := parentA.n.ActivationFn()
	if util.RandFloat(0, 1) < 0.5 {
		newActivationFn = parentB.n.ActivationFn()
	}
	newAggregationFn := parentA.n.AggregationFn()
	if util.RandFloat(0, 1) < 0.5 {
		newAggregationFn = parentB.n.AggregationFn()
	}

	newConn := net.NewNode(parentA.ID(), newBias, newActivationFn, newAggregationFn)
	return NewNodeGene(newConn), nil
}

type NodeGene struct {
	n *net.Node
}

func (g *NodeGene) ID() int64 {
	return g.n.ID()
}

func (g *NodeGene) Node() *net.Node {
	return g.n
}

func (g *NodeGene) Activate(inputs []float64, weights []float64) float64 {
	return g.n.Activate(inputs, weights)
}

func (g *NodeGene) Copy() *NodeGene {
	nodeCp := net.NewNode(g.n.ID(), g.n.Bias(), g.n.ActivationFn(), g.n.AggregationFn())
	nodeCp.SetActivation(g.n.Activation())
	return NewNodeGene(nodeCp)
}

func (g *NodeGene) Mutate(cfg *Config) {
	g.mutateBias(cfg)
	g.mutateActivationFn(cfg)
	g.mutateAggregationFn(cfg)
}

func (g *NodeGene) mutateBias(cfg *Config) {
	if util.RandFloat(0, 1) > cfg.BiasMutateRate {
		return
	}
	if util.RandFloat(0, 1) <= cfg.BiasReplaceRate {
		// We should replace the bias entirely
		g.n.SetBias(util.RandFloat(cfg.BiasMinValue, cfg.BiasMaxValue))
	} else {
		// We should adjust the bias by power
		lowBound := g.n.Bias() - cfg.BiasMutatePower
		if lowBound < cfg.BiasMinValue {
			lowBound = cfg.BiasMinValue
		}
		highBound := g.n.Bias() + cfg.BiasMutatePower
		if highBound > cfg.BiasMaxValue {
			highBound = cfg.BiasMaxValue
		}
		g.n.SetBias(util.RandFloat(lowBound, highBound))
	}
}

func (g *NodeGene) mutateActivationFn(cfg *Config) {
	if util.RandFloat(0, 1) > cfg.ActivationMutateRate {
		return
	}
	// TODO: Use cfg.ActivationOptions
	g.n.SetActivationFn(activation.RandFn())
}

func (g *NodeGene) mutateAggregationFn(cfg *Config) {
	if util.RandFloat(0, 1) > cfg.AggregationMutateRate {
		return
	}
	// TODO: Use cfg.AggregationOptions
	g.n.SetAggregationFn(aggregation.RandFn())
}

func (g *NodeGene) Distance(other *NodeGene, cfg *Config) float64 {
	d := math.Abs(g.n.Bias()-other.n.Bias()) + math.Abs(g.n.Activation()-other.n.Activation())
	myActivationFn := g.n.ActivationFn()
	otherActivationFn := other.n.ActivationFn()
	if &myActivationFn != &otherActivationFn {
		d += 1
	}

	myAggregationFn := g.n.AggregationFn()
	otherAggregationFn := other.n.AggregationFn()
	if &myAggregationFn != &otherAggregationFn {
		d += 1
	}

	return d * cfg.CompatibilityWeightCoefficient
}
