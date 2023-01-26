An Example C Project Using onnxruntime
======================================

I wanted an ONNX runtime example that uses pure C, but without all the slop
in the official pure C example.  Additionally, I wanted something that works
with mingw, so I don't use onnxruntime.lib. Basically, this depends on
onnxruntime.dll and the associated headers.

This example just runs a tiny network dumped from PyTorch.


Usage
-----

First, ensure that gcc is available on your PATH.

Next, run `buid_windows.bat` from this directory.

Finally, run `onnxruntime_example.exe` from this directory. (It will attempt to
load onnxruntime.dll from the `onnxruntime\lib` subdirectory.)


The Example Network
-------------------

The example neural network was generated using the `generate_network.py`
script. To generate a new network, simply run the script and wait for it to
complete.  However, you'll need to change the hardcoded `test_input` and
`expected_output` portions in `onnxruntime_example.c` to match the values
printed at the end of the script.

The network simply takes a vector of 4 values and returns a vector
approximating two values: the sum of all 4 inputs, and the maximum difference
between any two of the inputs.

