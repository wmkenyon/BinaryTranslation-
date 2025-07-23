# BinaryTranslation-
The goal of this project is to translate x86 binary to another ISA like RISC-V to make a processor with a different ISA x86-compliant without needing a cross-compiler.

## How I plan on doing it:
I am working on implementing two algorithms:
* One with a large neural network
* One using a transformer and some kind of attention mechanism

### Workflow (I guess???):
* Implement and test both algorithms using Pytorch
* Use a supervised learning method with a Linux kernel compiled for both x86 and RISC-V binaries
* Train the model
* Validate by adding new software to the compiled images

## Where would I like to be by the end of this
Ideally, I would get translations with accuracy in the high 90s.
Test it on an iGPU, or at least work out the math of "will this work on an iGPU with much less FLOPs".
_MAYBE_ test it using the AMD Ryzen NPU.
