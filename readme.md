#   MicroGPT-C
A C port of Andrej Karpathy's [MicroGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), written without any dependencies beyond the standard library. 

Compared to the original, the current microgpt-c has a ~32x speedup on average time-per-epoch over 1000 epochs


```
Performance counter stats for './main':

             0      context-switches:u               #      0.0 cs/sec  cs_per_second     
             0      cpu-migrations:u                 #      0.0 migrations/sec  migrations_per_second
       673,913      page-faults:u                    # 216895.2 faults/sec  page_faults_per_second
      3,107.09 msec task-clock:u                     #      1.0 CPUs  CPUs_utilized       
    18,532,638      cpu_core/branch-misses/u         #      0.5 %  branch_miss_rate         (99.81%)
 4,033,483,079      cpu_core/branches/u              #   1298.2 M/sec  branch_frequency     (99.81%)
 6,912,081,802      cpu_core/cpu-cycles/u            #      2.2 GHz  cycles_frequency       (99.81%)
19,537,667,088      cpu_core/instructions/u          #      2.8 instructions  insn_per_cycle  (99.81%)
    10,118,524      cpu_atom/branch-misses/u         #      0.3 %  branch_miss_rate         (0.03%)
 4,632,166,754      cpu_atom/branches/u              #   1490.8 M/sec  branch_frequency     (0.03%)
 6,998,143,681      cpu_atom/cpu-cycles/u            #      2.3 GHz  cycles_frequency       (0.06%)
12,501,881,248      cpu_atom/instructions/u          #      2.2 instructions  insn_per_cycle  (0.10%)
         TopdownL1 (cpu_core)                        #      4.8 %  tma_bad_speculation    
                                                     #     17.2 %  tma_frontend_bound       (99.81%)
                                                     #     32.5 %  tma_backend_bound      
                                                     #     45.5 %  tma_retiring             (99.81%)
         TopdownL1 (cpu_atom)                        #     34.0 %  tma_backend_bound        (0.16%)
                                                     #     13.8 %  tma_frontend_bound       (0.13%)
                                                     #     16.2 %  tma_bad_speculation    
                                                     #     36.0 %  tma_retiring             (0.10%)

   3.231840422 seconds time elapsed

   2.188536000 seconds user
   1.024326000 seconds sys
```
















