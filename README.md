An Example C Project Using onnxruntime
======================================

I wanted an ONNX runtime example that uses pure C, but without all the slop
in the official pure C example.  Additionally, I wanted something that works
with mingw, so I don't use onnxruntime.lib. Basically, this depends on
onnxruntime.dll and the associated headers.

This example just runs a simple network I dumped from PyTorch.


Usage
-----

First, ensure that gcc is available on your PATH.

Next, run `buid_windows.bat` from this directory.

Finally, run `onnxruntime_example.exe` from this directory. (It will attempt to
load onnxruntime.dll from the `lib/` subdirectory.)

