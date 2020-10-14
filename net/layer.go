package net

import (
	"github.com/jmwri/neatgo/activation"
	"github.com/jmwri/neatgo/aggregation"
)

func NewLayerDefinition(numNodes int, biasInitMin float64, biasInitMax float64, activationFn activation.Fn, aggregateFn aggregation.Fn) LayerDefinition {
	return LayerDefinition{
		NumNodes:      numNodes,
		BiasInitMin:   biasInitMin,
		BiasInitMax:   biasInitMax,
		ActivationFn:  activationFn,
		AggregationFn: aggregateFn,
	}
}

type LayerDefinition struct {
	NumNodes      int
	BiasInitMin   float64
	BiasInitMax   float64
	ActivationFn  activation.Fn
	AggregationFn aggregation.Fn
}

type Layer []*Node

type LayerConnections []*Connection
