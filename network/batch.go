package network

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// BatchOutputs holds what a batch of networks produced for a batch of inputs:
// for every network, one output vector per input sample.
//
// It is one flat slice rather than a slice of slices of slices, because a
// population run over a dataset produces hundreds of thousands of vectors and
// allocating each one would cost more than computing it.
type BatchOutputs struct {
	// Data is laid out [network][sample][output].
	Data       []float64
	NumNets    int
	NumSamples int
	NumOutputs int
}

// NewBatchOutputs allocates zeroed outputs of the given shape.
func NewBatchOutputs(nets, samples, outputs int) BatchOutputs {
	return BatchOutputs{
		Data:       make([]float64, nets*samples*outputs),
		NumNets:    nets,
		NumSamples: samples,
		NumOutputs: outputs,
	}
}

// Net returns the outputs of network i. It aliases the batch's storage.
func (b BatchOutputs) Net(i int) Outputs {
	width := b.NumSamples * b.NumOutputs
	return Outputs{data: b.Data[i*width : (i+1)*width : (i+1)*width], samples: b.NumSamples, width: b.NumOutputs}
}

// Outputs is what one network produced for every sample of a batch.
type Outputs struct {
	data    []float64
	samples int
	width   int
}

// Len is the number of samples.
func (o Outputs) Len() int { return o.samples }

// Sample returns the output vector for sample i. It aliases the batch's storage
// and must not be modified.
func (o Outputs) Sample(i int) []float64 {
	return o.data[i*o.width : (i+1)*o.width : (i+1)*o.width]
}

// ActivateBatch runs every network over every input on the CPU, spreading the
// networks across parallelism goroutines (0 means one per CPU).
//
// It is the reference the GPU implementation is checked against, and the right
// choice for a batch too small for a GPU to pay for its own transfers. Every
// network must be feed-forward, and all of them must agree on the number of
// outputs; each input must have the size every network expects.
func ActivateBatch(nets []*Network, inputs [][]float64, parallelism int) (BatchOutputs, error) {
	numOutputs, err := CheckBatch(nets, inputs)
	if err != nil {
		return BatchOutputs{}, err
	}
	result := NewBatchOutputs(len(nets), len(inputs), numOutputs)
	if len(nets) == 0 || len(inputs) == 0 {
		return result, nil
	}

	workers := parallelism
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	workers = min(workers, len(nets))

	var (
		next     atomic.Int64
		wg       sync.WaitGroup
		failOnce sync.Once
		failure  error
	)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= len(nets) {
					return
				}
				out := result.Net(i)
				for s, input := range inputs {
					if err := nets[i].ActivateInto(input, out.data[s*numOutputs:(s+1)*numOutputs]); err != nil {
						failOnce.Do(func() { failure = err })
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	return result, failure
}
