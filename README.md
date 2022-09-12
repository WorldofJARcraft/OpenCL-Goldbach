# Prime Sieve Implementation in OpenCL

## Overview
This program numerically analyses two variations of Goldbach's conjecture using OpenCL.  
The first variation states that any odd number can be represented as the sum of a prime number and the double of a squared integer larger than 0.  
The second variation allows the two squared integers to be different.  
The implementation uses the sieve of Eratosthenes, using one work item per factor, to find prime numbers.  
Given the prime numbers, two similar sieves are used to test the hypotheses.  

## Building

Build using CMake:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

On Windows, execute `Debug\OpenCLSieve.exe`.  
On Linux and UNIX-like operating systems, execute bin/OpenCLSieve.