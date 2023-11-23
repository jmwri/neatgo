package activation

import (
	"errors"
	"sync"
)

var (
	DefaultRegistry = NewRegistry()
	ErrNotFound     = errors.New("function not found")
)

func NewRegistry() *Registry {
	return &Registry{
		mu:  &sync.RWMutex{},
		fns: make(map[string]Fn),
	}
}

type Registry struct {
	mu  *sync.RWMutex
	fns map[string]Fn
}

func (r *Registry) Set(fn Fn) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.fns[fn.Name()] = fn
}

func (r *Registry) Get(name string) (Fn, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	fn, ok := r.fns[name]
	if ok {
		return fn, nil
	}
	return fn, ErrNotFound
}
