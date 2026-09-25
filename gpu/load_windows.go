package gpu

import "syscall"

// openLibrary loads the CUDA driver, which ships with the NVIDIA display driver
// rather than with any toolkit.
func openLibrary() (uintptr, error) {
	handle, err := syscall.LoadLibrary("nvcuda.dll")
	return uintptr(handle), err
}
