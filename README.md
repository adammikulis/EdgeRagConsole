# EdgeRag
This is a C# project that uses only LLamaSharp to build a retrieval-augmented generative A.I. pipeline. You can get a local large language models (LLM) by downloading a compatible .gguf file from HuggingFace, putting it in the folder C:/ai/models or change the programmed path to your own. Start this by either running the .exe in the bin/release/net8.0 folder or run it through Visual Studio/VS Code.

This requires the nuget pacakge [LLamaSharp](https://www.nuget.org/packages/LLamaSharp) to be installed to the project solution with Visual Studio or [this extension](https://marketplace.visualstudio.com/items?itemName=aliasadidev.nugetpackagemanagergui) in VS Code.
If you only have a CPU, install the [LLamaSharp.Backend.Cpu](https://www.nuget.org/packages/LLamaSharp.Backend.Cpu) nuget package to the project solution 
If you have an Nvidia GPU with at least 4GB of VRAM, make sure you have [Cuda 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) installed and grab the [LLamaSharp.Backend.Cuda12](https://www.nuget.org/packages/LLamaSharp.Backend.Cuda12) package instead.
