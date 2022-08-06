package network

import "sync"

type IDProvider struct {
	mu      sync.Mutex
	current int
}

func (p *IDProvider) Next() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.current += 1
	return p.current
}