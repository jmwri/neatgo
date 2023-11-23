package util

import "sync"

func AggregateChannels[T any](inChans []<-chan T) <-chan T {
	wg := &sync.WaitGroup{}
	outCh := make(chan T)

	wg.Add(len(inChans))
	for _, inCh := range inChans {
		go func(inCh <-chan T) {
			defer wg.Done()
			for v := range inCh {
				outCh <- v
			}
		}(inCh)
	}

	go func() {
		wg.Wait()
		close(outCh)
	}()

	return outCh
}

func ReadAllChan[T any](inCh <-chan T) []T {
	all := make([]T, 0)
	for v := range inCh {
		all = append(all, v)
	}
	return all
}
