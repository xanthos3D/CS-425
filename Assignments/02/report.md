|         |            |             |               |                                                                       |   |   |   |   |   |
|---------|------------|-------------|---------------|-----------------------------------------------------------------------|---|---|---|---|---|
|         |            |             |               |                                                                       |   |   |   |   |   |
|         |            |             |               |                                                                       |   |   |   |   |   |
| version | Time(Test) | Time(Final) | SpeedUp(Test) | Changes                                                               |   |   |   |   |   |
| V1      | 1m46s      |             | 1.0x          | No Changes                                                            |   |   |   |   |   |
| V2      | 17s        |             | 6.2x          | Parallelized the program using threading techniques.                  |   |   |   |   |   |
| V3      | .644s      | 1m26s       | 158x          | g++ -std=c++20 lychrel.cpp -o lychrel -03. Used optimizer  –O3        |   |   |   |   |   |
| V4      | .477s      | 1m25s       | 237x          | Replaced for loop with while loop to allow for thread synchronization |   |   |   |   |   |
|         |            |             |               |                                                                       |   |   |   |   |   |
|         |            |             |               |                                                                       |   |   |   |   |   |