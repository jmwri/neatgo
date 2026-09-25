package gpu

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ErrUnavailable is returned when there is no usable GPU: the CUDA driver is not
// installed, reports no devices, or does not support this platform. It is the
// ordinary condition on a machine without an NVIDIA GPU, so callers are expected
// to test for it and fall back to the CPU.
var ErrUnavailable = errors.New("gpu: no CUDA device available")

// Error is a failed CUDA driver call.
type Error struct {
	// Op is the driver function that failed.
	Op string
	// Code is the CUresult it returned.
	Code int32
	// Name is the driver's name for the code, such as CUDA_ERROR_OUT_OF_MEMORY.
	Name string
	// Log is the driver's diagnostic output, when it gave any.
	Log string
}

func (e *Error) Error() string {
	msg := fmt.Sprintf("gpu: %s: %s (%d)", e.Op, e.Name, e.Code)
	if e.Log != "" {
		msg += ": " + e.Log
	}
	return msg
}

// driver is the handful of CUDA driver API entry points this package uses,
// bound at run time. Binding them from the driver's own library, instead of
// linking against the CUDA toolkit, is what keeps the package free of cgo.
type driver struct {
	cuInit                    func(flags uint32) int32
	cuDeviceGetCount          func(count *int32) int32
	cuDeviceGet               func(device *int32, ordinal int32) int32
	cuDeviceGetName           func(name *byte, length int32, device int32) int32
	cuDevicePrimaryCtxRetain  func(ctx *uintptr, device int32) int32
	cuDevicePrimaryCtxRelease func(device int32) int32
	cuCtxSetCurrent           func(ctx uintptr) int32
	cuCtxSynchronize          func() int32
	cuModuleLoadDataEx        func(module *uintptr, image *byte, numOptions uint32, options *int32, optionValues *uintptr) int32
	cuModuleUnload            func(module uintptr) int32
	cuModuleGetFunction       func(fn *uintptr, module uintptr, name string) int32
	cuMemAlloc                func(ptr *uint64, size uintptr) int32
	cuMemFree                 func(ptr uint64) int32
	cuMemcpyHtoD              func(dst uint64, src unsafe.Pointer, size uintptr) int32
	cuMemcpyDtoH              func(dst unsafe.Pointer, src uint64, size uintptr) int32
	cuMemGetInfo              func(free, total *uintptr) int32
	cuLaunchKernel            func(fn uintptr, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedBytes uint32, stream uintptr, params unsafe.Pointer, extra unsafe.Pointer) int32
	cuGetErrorName            func(code int32, name **byte) int32
}

var (
	loadOnce sync.Once
	loaded   *driver
	loadErr  error
)

// loadDriver binds the driver library once for the process.
func loadDriver() (*driver, error) {
	loadOnce.Do(func() {
		handle, err := openLibrary()
		if err != nil {
			loadErr = fmt.Errorf("%w: %v", ErrUnavailable, err)
			return
		}
		d := &driver{}
		// The _v2 names are what the CUDA headers map the unversioned ones to;
		// the unversioned symbols are the obsolete 32-bit-size variants.
		bindings := []struct {
			fn   any
			name string
		}{
			{&d.cuInit, "cuInit"},
			{&d.cuDeviceGetCount, "cuDeviceGetCount"},
			{&d.cuDeviceGet, "cuDeviceGet"},
			{&d.cuDeviceGetName, "cuDeviceGetName"},
			{&d.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain"},
			{&d.cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2"},
			{&d.cuCtxSetCurrent, "cuCtxSetCurrent"},
			{&d.cuCtxSynchronize, "cuCtxSynchronize"},
			{&d.cuModuleLoadDataEx, "cuModuleLoadDataEx"},
			{&d.cuModuleUnload, "cuModuleUnload"},
			{&d.cuModuleGetFunction, "cuModuleGetFunction"},
			{&d.cuMemAlloc, "cuMemAlloc_v2"},
			{&d.cuMemFree, "cuMemFree_v2"},
			{&d.cuMemcpyHtoD, "cuMemcpyHtoD_v2"},
			{&d.cuMemcpyDtoH, "cuMemcpyDtoH_v2"},
			{&d.cuMemGetInfo, "cuMemGetInfo_v2"},
			{&d.cuLaunchKernel, "cuLaunchKernel"},
			{&d.cuGetErrorName, "cuGetErrorName"},
		}
		for _, b := range bindings {
			if err := register(b.fn, handle, b.name); err != nil {
				loadErr = fmt.Errorf("%w: %v", ErrUnavailable, err)
				return
			}
		}
		if code := d.cuInit(0); code != 0 {
			loadErr = fmt.Errorf("%w: cuInit returned %s", ErrUnavailable, d.errorName(code))
			return
		}
		loaded = d
	})
	return loaded, loadErr
}

// register binds one symbol, turning purego's panic on a missing one into an
// error so an old driver reads as "unavailable" rather than a crash.
func register(fn any, handle uintptr, name string) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("symbol %s: %v", name, r)
		}
	}()
	purego.RegisterLibFunc(fn, handle, name)
	return nil
}

// errorName is the driver's name for a result code.
func (d *driver) errorName(code int32) string {
	var name *byte
	if d.cuGetErrorName(code, &name) != 0 || name == nil {
		return fmt.Sprintf("CUresult %d", code)
	}
	return goString(name)
}

// check turns a CUresult into an error.
func (d *driver) check(op string, code int32) error {
	if code == 0 {
		return nil
	}
	return &Error{Op: op, Code: code, Name: d.errorName(code)}
}

// goString copies a NUL-terminated C string.
func goString(p *byte) string {
	var n int
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(p, n))
}
