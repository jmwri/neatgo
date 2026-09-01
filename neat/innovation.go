package neat

import "sync"

// Innovations assigns NEAT historical markings.
//
// The point of a historical marking is that the *same* structural innovation
// gets the *same* ID no matter which genome discovers it. That is what lets
// crossover line two genomes up gene-for-gene, and what lets the compatibility
// distance tell "shares ancestry" apart from "happens to look similar". A
// plain incrementing counter cannot do this: two genomes that independently
// grow the same connection would be handed different IDs and would then look
// maximally dissimilar forever.
type Innovations struct {
	mu          sync.Mutex
	ids         IDProvider
	connections map[connectionKey]int
	splits      map[int]Split
}

type connectionKey struct {
	from, to int
}

// Split records the genes created when a connection is split by an add-node
// mutation, so that the same split always yields the same three IDs.
type Split struct {
	NodeID        int
	InConnection  int
	OutConnection int
}

func NewInnovations(ids IDProvider) *Innovations {
	return &Innovations{
		ids:         ids,
		connections: make(map[connectionKey]int),
		splits:      make(map[int]Split),
	}
}

// NodeID returns a brand new node ID with no structural meaning attached.
func (i *Innovations) NodeID() int {
	return i.ids.Next()
}

// ConnectionID returns the innovation number for a connection between two
// nodes, allocating one the first time this connection is ever seen.
func (i *Innovations) ConnectionID(from, to int) int {
	i.mu.Lock()
	defer i.mu.Unlock()
	key := connectionKey{from: from, to: to}
	if id, ok := i.connections[key]; ok {
		return id
	}
	id := i.ids.Next()
	i.connections[key] = id
	return id
}

// LookupSplit returns the genes previously allocated for splitting the given
// connection, without allocating any.
func (i *Innovations) LookupSplit(connectionID int) (Split, bool) {
	i.mu.Lock()
	defer i.mu.Unlock()
	split, ok := i.splits[connectionID]
	return split, ok
}

// SplitConnection returns the genes for splitting the given connection with a
// new node, allocating them the first time this connection is ever split.
func (i *Innovations) SplitConnection(connectionID, from, to int) Split {
	i.mu.Lock()
	defer i.mu.Unlock()
	if split, ok := i.splits[connectionID]; ok {
		return split
	}
	split := Split{
		NodeID:        i.ids.Next(),
		InConnection:  i.ids.Next(),
		OutConnection: i.ids.Next(),
	}
	i.splits[connectionID] = split
	// Register the two halves so that a later "add connection" mutation
	// producing the same edge reuses the same innovation number.
	i.connections[connectionKey{from: from, to: split.NodeID}] = split.InConnection
	i.connections[connectionKey{from: split.NodeID, to: to}] = split.OutConnection
	return split
}

// SetCurrentID moves the underlying ID sequence past n, so that genes created
// from here on will not collide with hand-built genomes using IDs up to n.
func (i *Innovations) SetCurrentID(n int) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.ids.SetCurrent(n)
}
