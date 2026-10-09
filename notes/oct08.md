dut
  F1 pick   yj614phc  eval_f1=0.004  eval_acc=0.004  cond=1.0
  cond pick 15uroeyq  eval_f1=0.002  eval_acc=0.002  cond=1.0
fre
  F1 pick   6okg216x  eval_f1=0.088  eval_acc=0.091  cond=0.975
  cond pick x7z81r2w  eval_f1=0.037  eval_acc=0.037  cond=1.0
geo
  F1 pick   jtdov1jn  eval_f1=0.137  eval_acc=0.233  cond=0.590
  cond pick qtgttq8v  eval_f1=0.002  eval_acc=0.002  cond=1.0
isl
  F1 pick   612l6w9n  eval_f1=0.552  eval_acc=0.717  cond=0.770
  cond pick yhqjvxu7  eval_f1=0.552  eval_acc=0.716  cond=0.770
spa
  F1 pick   1wofy6ko  eval_f1=0.589  eval_acc=0.851  cond=0.691
  cond pick 76z11gpe  eval_f1=0.542  eval_acc=0.666  cond=0.813
swe
  F1 pick   asew5y0m  eval_f1=0.552  eval_acc=0.568  cond=0.971
  cond pick vtpwgb88  eval_f1=0.551  eval_acc=0.566  cond=0.973


Conditional-accuracy selection: ties on isl/swe; degenerate on geo/dut/fre. Real trade-off only on spa (acc 0.85/cond 0.69 vs acc 0.67/cond 0.81). Needs a minimum-acceptance floor before it's usable. 