package id

import "sync"

func NewProvider() *Provider {
	return &Provider{
		mu:      &sync.Mutex{},
		current: 0,
	}
}

type Provider struct {
	mu      *sync.Mutex
	current int64
}

func (p *Provider) Next() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.current += 1
	return p.current
}
