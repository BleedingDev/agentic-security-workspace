This directory contain examples of mixed-mode executables:

## EmptyClass

These executables have been built with the C compiler `cl` and a netmodule file built with the C# compiler `csc` and linked together using the `CLRIMAGETYPE:IJW` flag. This produces a sample that some tools may assume is a .NET assembly has no code visible in dnSpy. Unlike `ModuleCode`, the two source projects used to build `EmptyClass` do not interact with each other. One artifact that can quickly alert someone to mixed mode without looking at any flags or headers is that `mainCRTStartup` is referenced, this would indicate mixed mode.

## Empty Class DLL

This project is very similar to Empty Class, it has displays a message box instead of printing to screen. The message box is displayed when calling the export Demo
```
rundll32.exe EmptyClass_amd64.dll,Demo
```
Because this is a DLL the entrypoint will be native, but the native code will call `corDllMain`. This means no `mainCRTStartup` is referenced, the binary in dnSpy really looks like it does nothing. Opening in IDA and specifying PE for the loader will easily show an export that can be called. Writing DllMain in unmanaged code will not be called.

## ModuleCode

This project is more complex to get setup, but allows for managed and unmanaged code to live in the same source file. The drawback is the source code is easily seen in dnSpy by looking under the Module class. 

## references

[The Wover's Mixed Assemblies](https://thewover.github.io/Mixed-Assemblies/)  
[Call managed code from your C++ code](https://docs.microsoft.com/en-us/archive/blogs/calvin_hsia/call-managed-code-from-your-c-code)
[CVE-2019-18935 using Mixed Assemblies for delivery](https://github.com/noperator/CVE-2019-18935)  
[Native and .NET Interoperability](https://docs.microsoft.com/en-us/cpp/dotnet/native-and-dotnet-interoperability?view=msvc-170)