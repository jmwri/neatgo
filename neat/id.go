package neat

import "sync"

type IDProvider interface {
	Next() int
}

func NewSequentialIDProvider() *SequentialIDProvider {
	return &SequentialIDProvider{
		mu:      sync.Mutex{},
		current: 0,
	}
}

type SequentialIDProvider struct {
	mu      sync.Mutex
	current int
}

func (p *SequentialIDProvider) Next() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.current += 1
	return p.current
}
