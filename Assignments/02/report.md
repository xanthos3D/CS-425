| Version | Time(test) | Time(Final) | Speedup(Final) | Memory(kb) | Changes                                                               |   |   |   |   |
|---------|------------|-------------|----------------|------------|-----------------------------------------------------------------------|---|---|---|---|
| V1      | 1m46 s     | 1.0x        | 74m26s         | 3680       | No Changes                                                            |   |   |   |   |
| V2      | 17s        |             |                |            | Parallelized the program using threading techniques.                  |   |   |   |   |
| V3      | 0.644s     | 52.9x       | 83.63s         | 4872       | g++ -std=c++20 lychrel.cpp -o lychrel -03. Used optimizer  –O3        |   |   |   |   |
| V4      | 0.477s     | 49.23x      | 89.90s         | 4936       | Replaced for loop with while loop to allow for thread synchronization |   |   |   |   |
|         |            |             |                |            |                                                                       |   |   |   |   |
|         |            |             |                |            |                                                                       |   |   |   |   |
|         |            |             |                |            |                                                                       |   |   |   |   |
|         |            |             |                |            |                                                                       |   |   |   |   |
|         |            |             |                |            |                                                                       |   |   |   |   |
