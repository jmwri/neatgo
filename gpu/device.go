// Package gpu runs batches of compiled networks on an NVIDIA GPU.
//
// It needs only the NVIDIA display driver, which provides the CUDA driver
// library: no CUDA toolkit, no C compiler, and no cgo, so a program using it
// still builds with a plain go build and simply reports ErrUnavailable at run
// time on a machine without a GPU.
//
// What a GPU can speed up here is narrow, and worth being clear about. A NEAT
// network is small and irregular - a few dozen nodes with no layers to turn into
// matrix products - so a GPU gains nothing from any single activation. It gains
// from volume: running every network of a population over every sample of a
// dataset is a great many independent, identical-shaped computations, and that is
// what ActivateBatch does in one launch. A task that has to be driven step by
// step, like a game, cannot be batched this way and stays on the CPU.
//
// The GPU computes in float32, using the hardware's approximate exp, log, sin and
// reciprocal, where the CPU path uses float64. Results agree to a small relative
// error (roughly 1e-5 for ordinary networks), not bit for bit, so a run's exact
// trajectory differs from the same seed run on the CPU. Each is reproducible
// against itself.
package gpu

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/jmwri/neatgo/v2/network"
)

// Available reports whether a CUDA device can be opened here.
func Available() bool {
	d, err := loadDriver()
	if err != nil {
		return false
	}
	var count int32
	return d.cuDeviceGetCount(&count) == 0 && count > 0
}

// blockSize is the number of threads per block. Every thread is independent, so
// this only needs to be a comfortable multiple of the warp size.
const blockSize = 128

// scratchCeiling caps the scratch buffer a launch may claim, in bytes, along
// with a quarter of the device's free memory. It exists so that a population over
// a huge dataset is split into several launches rather than exhausting the
// device.
const scratchCeiling = 1 << 30

// Device is an open CUDA device with the kernel loaded. It is safe for
// concurrent use; calls run one at a time, since they share the device's
// buffers.
type Device struct {
	mu      sync.Mutex
	drv     *driver
	ordinal int32
	name    string
	ctx     uintptr
	module  uintptr
	kernel  uintptr
	closed  bool
	nets    buffer
	nodes   buffer
	edges   buffer
	outIdx  buffer
	inputs  buffer
	values  buffer
	out     buffer
	hostOut []float32
	// scratchLimit, when set, replaces the scratch budget. Tests use it to force
	// a launch to be split.
	scratchLimit uintptr
	// launched counts the kernel launches made, so a test can tell a batch that
	// ran on the GPU from one that quietly fell back to the CPU.
	launched int
}

// Open opens the CUDA device with the given ordinal (0 is the first) and loads
// the kernel onto it. It returns an error wrapping ErrUnavailable if there is no
// such device, and an *Error if the driver refuses the kernel - most likely a GPU
// older than the Maxwell generation.
func Open(ordinal int) (*Device, error) {
	drv, err := loadDriver()
	if err != nil {
		return nil, err
	}
	var count int32
	if err := drv.check("cuDeviceGetCount", drv.cuDeviceGetCount(&count)); err != nil {
		return nil, err
	}
	if ordinal < 0 || int32(ordinal) >= count {
		return nil, fmt.Errorf("%w: device %d requested, %d present", ErrUnavailable, ordinal, count)
	}

	d := &Device{drv: drv}
	if err := drv.check("cuDeviceGet", drv.cuDeviceGet(&d.ordinal, int32(ordinal))); err != nil {
		return nil, err
	}
	var name [256]byte
	if err := drv.check("cuDeviceGetName", drv.cuDeviceGetName(&name[0], int32(len(name)), d.ordinal)); err != nil {
		return nil, err
	}
	d.name = goString(&name[0])

	// The primary context is the one the runtime API shares, so this coexists
	// with anything else in the process using the same GPU.
	if err := drv.check("cuDevicePrimaryCtxRetain", drv.cuDevicePrimaryCtxRetain(&d.ctx, d.ordinal)); err != nil {
		return nil, err
	}

	// A context is current for one OS thread at a time, so every operation that
	// uses it pins its goroutine to a thread first.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := d.setCurrent(); err != nil {
		d.release()
		return nil, err
	}

	// Ask the driver to describe a compile failure rather than only fail.
	logBuf := make([]byte, 8192)
	options := []int32{cuJitErrorLogBuffer, cuJitErrorLogBufferSizeBytes}
	values := []uintptr{uintptr(unsafe.Pointer(&logBuf[0])), uintptr(len(logBuf))}
	image := append([]byte(buildPTX()), 0)
	code := drv.cuModuleLoadDataEx(&d.module, &image[0], uint32(len(options)), &options[0], &values[0])
	runtime.KeepAlive(logBuf)
	if code != 0 {
		err := drv.check("cuModuleLoadDataEx", code).(*Error)
		err.Log = goString(&logBuf[0])
		d.release()
		return nil, err
	}
	if err := drv.check("cuModuleGetFunction", drv.cuModuleGetFunction(&d.kernel, d.module, kernelName)); err != nil {
		d.release()
		return nil, err
	}
	return d, nil
}

// CU_JIT_ERROR_LOG_BUFFER and its size, from cuda.h.
const (
	cuJitErrorLogBuffer          = 5
	cuJitErrorLogBufferSizeBytes = 6
)

// Name is the device's marketing name, such as "NVIDIA GeForce RTX 3090 Ti".
func (d *Device) Name() string { return d.name }

// Close releases the device's memory and kernel. The Device must not be used
// afterwards.
func (d *Device) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	d.closed = true
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := d.setCurrent(); err != nil {
		return err
	}
	for _, b := range []*buffer{&d.nets, &d.nodes, &d.edges, &d.outIdx, &d.inputs, &d.values, &d.out} {
		b.free(d.drv)
	}
	d.release()
	return nil
}

// release lets go of the module and context. Callers hold the context current.
func (d *Device) release() {
	if d.module != 0 {
		d.drv.cuModuleUnload(d.module)
		d.module = 0
	}
	if d.ctx != 0 {
		d.drv.cuDevicePrimaryCtxRelease(d.ordinal)
		d.ctx = 0
	}
}

func (d *Device) setCurrent() error {
	return d.drv.check("cuCtxSetCurrent", d.drv.cuCtxSetCurrent(d.ctx))
}

// buffer is device memory that only ever grows, so that a run which activates a
// population every generation allocates on the first and reuses ever after.
type buffer struct {
	ptr  uint64
	size uintptr
}

func (b *buffer) ensure(d *driver, size uintptr) error {
	if size <= b.size {
		return nil
	}
	b.free(d)
	// A zero-size allocation is an error to the driver; a word is always fine.
	size = max(size, 16)
	if err := d.check("cuMemAlloc", d.cuMemAlloc(&b.ptr, size)); err != nil {
		return err
	}
	b.size = size
	return nil
}

func (b *buffer) free(d *driver) {
	if b.ptr != 0 {
		d.cuMemFree(b.ptr)
		b.ptr, b.size = 0, 0
	}
}

// upload copies a host slice to the front of the buffer, growing it if needed.
func upload[T any](d *Device, b *buffer, data []T) error {
	size := uintptr(len(data)) * unsafe.Sizeof(*new(T))
	if err := b.ensure(d.drv, size); err != nil {
		return err
	}
	if size == 0 {
		return nil
	}
	return d.drv.check("cuMemcpyHtoD", d.drv.cuMemcpyHtoD(b.ptr, unsafe.Pointer(&data[0]), size))
}

// kernelArgs mirrors the kernel's parameter list. The driver is handed a pointer
// to each field, so it lives on the heap.
type kernelArgs struct {
	nets, nodes, edges, outIdx, inputs, values, out uint64
	numNets, samples, numIn, numOut                 uint32
}

// ActivateBatch runs every network over every input and returns each network's
// output for each sample: the same result as network.ActivateBatch, computed on
// the GPU in float32.
//
// Every network must be feed-forward and all must have the same number of
// outputs. A network using an activation function the kernel does not implement -
// one registered by the caller - runs on the CPU instead, alongside, so a custom
// function costs speed rather than correctness.
func (d *Device) ActivateBatch(nets []*network.Network, inputs [][]float64) (network.BatchOutputs, error) {
	numOutputs, err := network.CheckBatch(nets, inputs)
	if err != nil {
		return network.BatchOutputs{}, err
	}
	result := network.NewBatchOutputs(len(nets), len(inputs), numOutputs)
	if len(nets) == 0 || len(inputs) == 0 || numOutputs == 0 {
		return result, nil
	}

	var enc encoded
	var onGPU, onCPU []int
	for i, net := range nets {
		program, err := net.Program()
		if err != nil {
			return network.BatchOutputs{}, err
		}
		if canRun(program) {
			enc.add(program)
			onGPU = append(onGPU, i)
		} else {
			onCPU = append(onCPU, i)
		}
	}

	if len(onCPU) > 0 {
		cpuNets := make([]*network.Network, len(onCPU))
		for k, i := range onCPU {
			cpuNets[k] = nets[i]
		}
		cpu, err := network.ActivateBatch(cpuNets, inputs, 0)
		if err != nil {
			return network.BatchOutputs{}, err
		}
		width := len(inputs) * numOutputs
		for k, i := range onCPU {
			copy(result.Data[i*width:(i+1)*width], cpu.Data[k*width:(k+1)*width])
		}
	}
	if len(onGPU) == 0 {
		return result, nil
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return network.BatchOutputs{}, fmt.Errorf("gpu: device is closed")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := d.setCurrent(); err != nil {
		return network.BatchOutputs{}, err
	}
	if err := d.run(enc, onGPU, inputs, result); err != nil {
		return network.BatchOutputs{}, err
	}
	return result, nil
}

// run launches the kernel for the encoded networks, in as many pieces as the
// scratch budget needs, and scatters the results into result.
func (d *Device) run(enc encoded, onGPU []int, inputs [][]float64, result network.BatchOutputs) error {
	drv := d.drv
	numNets, samples := len(onGPU), len(inputs)
	numIn, numOut := len(inputs[0]), result.NumOutputs

	flat := make([]float32, samples*numIn)
	for s, input := range inputs {
		for j, x := range input {
			flat[s*numIn+j] = float32(x)
		}
	}
	for _, step := range []struct {
		name string
		do   func() error
	}{
		{"nets", func() error { return upload(d, &d.nets, enc.nets) }},
		{"nodes", func() error { return upload(d, &d.nodes, enc.nodes) }},
		{"edges", func() error { return upload(d, &d.edges, enc.edges) }},
		{"outputs", func() error { return upload(d, &d.outIdx, enc.outIdx) }},
		{"inputs", func() error { return upload(d, &d.inputs, flat) }},
	} {
		if err := step.do(); err != nil {
			return err
		}
	}

	// Each launch needs maxNodes floats of scratch per thread, and its thread
	// count must keep every index it computes inside 32 bits.
	var free, total uintptr
	budget := uintptr(scratchCeiling)
	if drv.cuMemGetInfo(&free, &total) == 0 {
		budget = min(budget, free/4)
	}
	if d.scratchLimit != 0 {
		budget = d.scratchLimit
	}
	maxThreads := min(int(budget/4)/enc.maxNodes, math.MaxInt32/numOut)
	if maxThreads < 1 {
		return fmt.Errorf("gpu: a network of %d nodes does not fit in the device's scratch budget", enc.maxNodes)
	}
	sampleChunk := min(samples, maxThreads)
	netChunk := min(numNets, max(1, maxThreads/sampleChunk))

	args := &kernelArgs{
		nets: d.nets.ptr, nodes: d.nodes.ptr, edges: d.edges.ptr, outIdx: d.outIdx.ptr,
		numIn: uint32(numIn), numOut: uint32(numOut),
	}
	params := [11]unsafe.Pointer{
		unsafe.Pointer(&args.nets), unsafe.Pointer(&args.nodes), unsafe.Pointer(&args.edges),
		unsafe.Pointer(&args.outIdx), unsafe.Pointer(&args.inputs), unsafe.Pointer(&args.values),
		unsafe.Pointer(&args.out), unsafe.Pointer(&args.numNets), unsafe.Pointer(&args.samples),
		unsafe.Pointer(&args.numIn), unsafe.Pointer(&args.numOut),
	}

	if err := d.values.ensure(drv, uintptr(enc.maxNodes*netChunk*sampleChunk)*4); err != nil {
		return err
	}
	if err := d.out.ensure(drv, uintptr(netChunk*sampleChunk*numOut)*4); err != nil {
		return err
	}
	args.values, args.out = d.values.ptr, d.out.ptr

	for n0 := 0; n0 < numNets; n0 += netChunk {
		nc := min(netChunk, numNets-n0)
		for s0 := 0; s0 < samples; s0 += sampleChunk {
			sc := min(sampleChunk, samples-s0)
			args.nets = d.nets.ptr + uint64(n0*netRecordInts*4)
			args.inputs = d.inputs.ptr + uint64(s0*numIn*4)
			args.numNets, args.samples = uint32(nc), uint32(sc)

			threads := nc * sc
			blocks := uint32((threads + blockSize - 1) / blockSize)
			d.launched++
			code := drv.cuLaunchKernel(d.kernel, blocks, 1, 1, blockSize, 1, 1, 0, 0, unsafe.Pointer(&params[0]), nil)
			runtime.KeepAlive(args)
			if err := drv.check("cuLaunchKernel", code); err != nil {
				return err
			}
			if err := drv.check("cuCtxSynchronize", drv.cuCtxSynchronize()); err != nil {
				return err
			}

			count := nc * sc * numOut
			if cap(d.hostOut) < count {
				d.hostOut = make([]float32, count)
			}
			host := d.hostOut[:count]
			if err := drv.check("cuMemcpyDtoH", drv.cuMemcpyDtoH(unsafe.Pointer(&host[0]), d.out.ptr, uintptr(count)*4)); err != nil {
				return err
			}
			for k := 0; k < nc; k++ {
				dst := result.Data[(onGPU[n0+k]*samples+s0)*numOut:]
				src := host[k*sc*numOut : (k+1)*sc*numOut]
				for j, v := range src {
					dst[j] = float64(v)
				}
			}
		}
	}
	return nil
}
