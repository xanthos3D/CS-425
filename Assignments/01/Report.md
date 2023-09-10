|version  |Time  |speedup|Memory(KB)| Changes                                                              | Command Used                                 |
------------------------------------------------------------------------------------------------------------------------------------------------------------
|01       |10.64s|1.0x   |1036536   | no changes yet                                                       |g++ -std=c++20 01.cpp -o 01                   |
|02       |10.00s|1.06x  |1040888   | slight speed up.                                                     |g++ -std=c++20 01.cpp -o 01 - o0              |
|03       |9.72s |1.09x  |1041208   | little more speed up and more memory usage                           |g++ -std=c++20 01.cpp -o 01 - o1              |
|04       |10.10s|1.05x  |1040788   | not as much speed up but slightly less memory used than other options|g++ -std=c++20 01.cpp -o 01 - o2              |
|05       |9.94s |1.07x  |1040760   | similiar speed up and memory usage as 02                             |g++ -std=c++20 01.cpp -o 01 - o3              |
|06       |10.02s|1.06x  |1040884   | relatively similiar to 02 and 05 inters of time and memory usage.    |g++ -std=c++20 01.cpp -o 01 - og              |
|07       |11.44s|0.92x  |1040856   | speed decreased an memory stayed the same                            |g++ -std=c++20 01.cpp -o 01 -fprofile-arcs    |
|08       |9.82s |1.08x  |1040944   | speed up, but memory usage is similiar to previous numbers.          |g++ -std=c++20 01.cpp -o 01 -funroll-all-loops|