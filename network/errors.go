package network

import "errors"

// Errors returned by Compile and by a compiled network.
//
// These are distinguishable with errors.Is so a caller can tell a genome that
// is merely malformed from one that asks for something the process does not
// have registered, and react differently - a cycle is a bug in whatever built
// the genome, an unknown activation is usually a missing registration at
// startup.
var (
	// ErrCycle is returned when the enabled connections form a loop, which has
	// no feed-forward evaluation order.
	ErrCycle = errors.New("network: contains a cycle")
	// ErrUnknownActivation is returned when a node names an activation
	// function that is not in the registry.
	ErrUnknownActivation = errors.New("network: unknown activation function")
	// ErrDuplicateNode is returned when two nodes share an ID.
	ErrDuplicateNode = errors.New("network: duplicate node id")
	// ErrInputSize is returned when the input does not match the number of
	// input nodes.
	ErrInputSize = errors.New("network: wrong number of inputs")
	// ErrOutputSize is returned when an ActivateInto buffer is the wrong size.
	ErrOutputSize = errors.New("network: wrong output buffer size")
)
