generate() also sends input symbols never seen in training to the "" arc. The expanded product only knows the training alphabet.

- Flex 4: "" in delta/psi = consumed default arc (generate() lines 53-81), not epsilon. expand_defaults() makes them explicit. Old 4 datasets: sizes unchanged vs August (no defaults used). swe/spa/isl now convert: trimmed/min = 84536/10653, 1995/1873, 84531/8292.
- swe/spa/isl conversions use yesterday's relaunch runs, not August's (August's had no artifacts).