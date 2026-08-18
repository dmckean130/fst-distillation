# (Aug 18 notes)

Inflection + bimachine is unreachable in the current code. Fix: implement `sample_full_domain` for the bimachine path.

Epsilon transitions block serialization on 3 datasets. Not fixed by `epsilon_remove()`. Fix: extend `delta` to admit "" as a symbol key and handle epsilon in the product.

bgpu-g6-u25 is unusable, sm 120 “no kernel image is available for execution".

Weighting by transition frequency will give a more accurate RCD number - helps fix hapax row firing especially in sparse data sets. 
