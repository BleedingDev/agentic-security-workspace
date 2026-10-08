This directory contains a number of .NET modules with unusual string constants:

  - `unpaired-surrogate`: references a string constant with an unpaired UTF-16 surrogate
  - `raw-binary`: references a string constant containing a raw binary MZ header
  - `string-decoder`: uses a routine to XOR decode a string constant at runtime

Some related references here:
https://github.com/xamarin/xamarin-android/issues/4909

> Normally this might not be a problem, but Dotfuscator takes advantage of a technical fact of how the .NET runtime treats user strings (those stored in the assembly's #US blob heap). Specifically, while strings are stored as sequences of UTF-16 code units, the runtime allows any sequence of 16-bit values, not just a sequence that is a well-formed UTF-16 string. For instance, .NET (and the C# compiler) allow strings to have unpaired UTF-16 surrogates. Dotfuscator's knowledge of this fact results in scenarios in protected code similar to the minimal repro: an ill-formed UTF-16 string constant is passed to one method, which returns a well-formed UTF-16 string (only known at runtime) for use in reflection.

and reference here: https://codeblog.jonskeet.uk/2014/11/07/when-is-a-string-not-a-string/
