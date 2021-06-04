# go-pytorch

[![Build Status](https://dev.azure.com/yhchang/c3sr/_apis/build/status/c3sr.go-pytorch?branchName=master)](https://dev.azure.com/yhchang/c3sr/_build/latest?definitionId=2&branchName=master)
[![Go Report Card](https://goreportcard.com/badge/github.com/c3sr/go-pytorch)](https://goreportcard.com/report/github.com/c3sr/go-pytorch)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Go binding for Pytorch C++ API.
This is used by the [Pytorch agent](https://github.com/c3sr/pytorch) in [MLModelScope](mlmodelscope.org) to perform model inference in Go.

## Installation

The binding requires Pytorch C++ (libtorch) and other Go packages.

### Pytorch C++ (libtorch) Library

The Pytorch C++ library is expected to be under `/opt/libtorch`.

The binding is built using libtorch 1.8.1.

To install Pytorch C++ on your system, you can

1. download pre-built binary from [Pytorch website](https://pytorch.org): Choose `Pytorch Build = Stable`, `Your OS = <fill>`, `Package = LibTorch`, `Language = C++` and `CUDA = <fill>`. Then download `cxx11 ABI` version. Unzip the packaged directory and copy to `/opt/libtorch` (or modify the corresponding `CFLAGS` and `LDFLAGS` paths if using a custom location).

2. build it from source: Refer to our [scripts](scripts) or the `LIBRARY INSTALLATION` section in the [dockefiles](dockerfiles).

- The default blas is OpenBLAS.
  The default OpenBLAS path for macOS is `/usr/local/opt/openblas` if installed throught homebrew (openblas is keg-only, which means it was not symlinked into /usr/local, because macOS provides BLAS and LAPACK in the Accelerate framework).

- The default pytorch C++ installation path is `/opt/libtorch` for linux, darwin and ppc64le without powerai

- The default CUDA path is `/usr/local/cuda`

See [lib.go](lib.go) for details.

If you get an error about not being able to write to `/opt` then perform the following

```
sudo mkdir -p /opt/libtorch
sudo chown -R `whoami` /opt/libtorch
```

If you are using Pytorch docker images or other libary paths, change CGO_CFLAGS, CGO_CXXFLAGS and CGO_LDFLAGS enviroment variables. Refer to [Using cgo with the go command](https://golang.org/cmd/cgo/#hdr-Using_cgo_with_the_go_command).

For example,

```
    export CGO_CFLAGS="${CGO_CFLAGS} -I/tmp/libtorch/include"
    export CGO_CXXFLAGS="${CGO_CXXFLAGS} -I/tmp/libtorch/include"
    export CGO_LDFLAGS="${CGO_LDFLAGS} -L/tmp/libtorch/lib"
```

There is [an issue](https://github.com/pytorch/pytorch/issues/27971) when using libtorch with version < 1.6.0, the work around here is to set `LRU_CACHE_CAPACITY=1` in the environmental variable.

### Configure Environmental Variables

Configure the linker environmental variables since the Pytorch C++ library is under a non-system directory. Place the following in either your `~/.bashrc` or `~/.zshrc` file

Linux
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/libtorch/lib
export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/libtorch/lib

```

macOS
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/libtorch/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/libtorch/lib
```
## Check the Build

Run `go build` to check the dependences installation and library paths set-up.
On linux, the default is to use GPU, if you don't have a GPU, do `go build -tags=nogpu` instead of `go build`.

**_Note_** : The CGO interface passes go pointers to the C API. This is an error by the CGO runtime. Disable the error by placing

```
export GODEBUG=cgocheck=0
```

in your `~/.bashrc` or `~/.zshrc` file and then run either `source ~/.bashrc` or `source ~/.zshrc`

## Credits

Parts of the implementation have been borrowed from [orktes/go-torch](https://github.com/orktes/go-torch)
