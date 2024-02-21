# EdgeRag
This is a C# project that uses only LLamaSharp to build a retrieval-augmented generative A.I. pipeline. Start this by either running the standalone .exe (no .NET required) or run it through Visual Studio/VS Code (.NET 8 required). The program helps you download a compatible large language model file (.gguf), which puts it in an easily accessible project folder called models (or change the programmed path to your own). 

Two builds are distributed, one is for computers with Nvidia GPUs (EdgeRagCuda12) and the other is for CPU-only (EdgeRag). Choose the version that applies to your hardware situation, and within Visual Studio you can test/release different versions. If you have an Nvidia GPU with at least 4GB of VRAM, make sure you have [Cuda 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) installed prior to using this software.

A sample database of synthetic tickets is provided (datasets/sampleDB.json) but the user can create and use as many others as they like. A high quality model (Mistral-7B-v0.2-Instruct at q8) was used to generate the tickets, and a lower quality one (of the Mistral faimly) can be used to parse through it.

This current iteration is a console application made entirely within Visual Studio, but future development is focused on a GUI version of EdgeRag that uses the Godot game engine.
