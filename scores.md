
Test: batch=32, chunk=16, length=64
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                   Test metric                   ┃                  DataLoader 0                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       activity_selector/output_score_test       │               0.7906249761581421                │
│      articulation_points/output_score_test      │                       0.0                       │
│         bellman_ford/output_score_test          │                   0.97265625                    │
│              bfs/output_score_test              │                    0.9765625                    │
│         binary_search/output_score_test         │                0.590624988079071                │
│            bridges/output_score_test            │              0.014691161923110485               │
│          bubble_sort/output_score_test          │                   0.98046875                    │
│      dag_shortest_paths/output_score_test       │                0.985546886920929                │
│              dfs/output_score_test              │                   0.97265625                    │
│           dijkstra/output_score_test            │                   0.97265625                    │
│ find_maximum_subarray_kadane/output_score_test  │               0.20624999701976776               │
│        floyd_warshall/output_score_test         │                     0.96875                     │
│          graham_scan/output_score_test          │               0.7152343988418579                │
│           heapsort/output_score_test            │                   0.97265625                    │
│        insertion_sort/output_score_test         │                    0.9765625                    │
│         jarvis_march/output_score_test          │               0.7515624761581421                │
│          kmp_matcher/output_score_test          │               0.15000000596046448               │
│          lcs_length/output_score_test           │                   0.853515625                   │
│      matrix_chain_order/output_score_test       │                    0.9921875                    │
│            minimum/output_score_test            │               0.3499999940395355                │
│          mst_kruskal/output_score_test          │                       0.0                       │
│           mst_prim/output_score_test            │                     0.96875                     │
│     naive_string_matcher/output_score_test      │               0.06875000149011612               │
│          optimal_bst/output_score_test          │                    0.9921875                    │
│          quickselect/output_score_test          │              0.0062500000931322575              │
│           quicksort/output_score_test           │               0.9781249761581421                │
│      segments_intersect/output_score_test       │                0.813671886920929                │
│ strongly_connected_components/output_score_test │                    0.9765625                    │
│        task_scheduling/output_score_test        │                0.842968761920929                │
│       topological_sort/output_score_test        │               0.49687498807907104               │
│             total/output_score_test             │               0.6779115796089172                │
└─────────────────────────────────────────────────┴─────────────────────────────────────────────────┘


Test: batch=64, chunk=16, length=64

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                   Test metric                   ┃                  DataLoader 0                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       activity_selector/output_score_test       │                0.821484386920929                │
│      articulation_points/output_score_test      │                       0.0                       │
│         bellman_ford/output_score_test          │                   0.97265625                    │
│              bfs/output_score_test              │                    0.9765625                    │
│         binary_search/output_score_test         │               0.44999998807907104               │
│            bridges/output_score_test            │              0.0005847931024618447              │
│          bubble_sort/output_score_test          │               0.9800781011581421                │
│      dag_shortest_paths/output_score_test       │               0.9859374761581421                │
│              dfs/output_score_test              │                     0.96875                     │
│           dijkstra/output_score_test            │                   0.97265625                    │
│ find_maximum_subarray_kadane/output_score_test  │               0.20781250298023224               │
│        floyd_warshall/output_score_test         │                     0.96875                     │
│          graham_scan/output_score_test          │               0.8363281488418579                │
│           heapsort/output_score_test            │                   0.97265625                    │
│        insertion_sort/output_score_test         │                   0.97265625                    │
│         jarvis_march/output_score_test          │                0.583203136920929                │
│          kmp_matcher/output_score_test          │               0.11406250298023224               │
│          lcs_length/output_score_test           │                   0.84765625                    │
│      matrix_chain_order/output_score_test       │                    0.9921875                    │
│            minimum/output_score_test            │               0.40156251192092896               │
│          mst_kruskal/output_score_test          │                       0.0                       │
│           mst_prim/output_score_test            │                     0.96875                     │
│     naive_string_matcher/output_score_test      │               0.06562499701976776               │
│          optimal_bst/output_score_test          │                    0.9921875                    │
│          quickselect/output_score_test          │               0.07656250149011612               │
│           quicksort/output_score_test           │                    0.9765625                    │
│      segments_intersect/output_score_test       │                0.821093738079071                │
│ strongly_connected_components/output_score_test │                    0.9765625                    │
│        task_scheduling/output_score_test        │                   0.822265625                   │
│       topological_sort/output_score_test        │                    0.484375                     │
│             total/output_score_test             │               0.6736522912979126                │
└─────────────────────────────────────────────────┴─────────────────────────────────────────────────┘

Test: batch=16, chunk=16, length=64 (Stacked)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                   Test metric                   ┃                  DataLoader 0                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       activity_selector/output_score_test       │               0.8812500238418579                │
│      articulation_points/output_score_test      │                       0.0                       │
│         bellman_ford/output_score_test          │                   0.97265625                    │
│              bfs/output_score_test              │                    0.9765625                    │
│         binary_search/output_score_test         │               0.6468750238418579                │
│            bridges/output_score_test            │               0.03709716722369194               │
│          bubble_sort/output_score_test          │               0.9769531488418579                │
│      dag_shortest_paths/output_score_test       │                0.985546886920929                │
│              dfs/output_score_test              │                   0.97265625                    │
│           dijkstra/output_score_test            │                   0.97265625                    │
│ find_maximum_subarray_kadane/output_score_test  │               0.2562499940395355                │
│        floyd_warshall/output_score_test         │                   0.97265625                    │
│          graham_scan/output_score_test          │                0.567187488079071                │
│           heapsort/output_score_test            │                   0.97265625                    │
│        insertion_sort/output_score_test         │                   0.97265625                    │
│         jarvis_march/output_score_test          │                0.862109363079071                │
│          kmp_matcher/output_score_test          │                    0.171875                     │
│          lcs_length/output_score_test           │                0.858593761920929                │
│      matrix_chain_order/output_score_test       │                    0.9921875                    │
│            minimum/output_score_test            │               0.3968749940395355                │
│          mst_kruskal/output_score_test          │                       0.0                       │
│           mst_prim/output_score_test            │                     0.96875                     │
│     naive_string_matcher/output_score_test      │               0.13750000298023224               │
│          optimal_bst/output_score_test          │                    0.9921875                    │
│          quickselect/output_score_test          │                     0.03125                     │
│           quicksort/output_score_test           │                   0.97265625                    │
│      segments_intersect/output_score_test       │               0.8433593511581421                │
│ strongly_connected_components/output_score_test │                    0.9765625                    │
│        task_scheduling/output_score_test        │                0.840624988079071                │
│       topological_sort/output_score_test        │                    0.484375                     │
│             total/output_score_test             │               0.6697872281074524                │
└─────────────────────────────────────────────────┴─────────────────────────────────────────────────┘
