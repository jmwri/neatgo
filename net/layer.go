package net

import "neatgo/activation"

type LayerDefinition struct {
	NumNodes int
	ActivationFn activation.Fn
}

type Layer []*Node

type LayerConnections []*Connection
