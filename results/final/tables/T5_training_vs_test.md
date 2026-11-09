## T5. Training vs inner-CV / validation vs test BA (P5 §5)

| Method | Protocol | Training BA | Inner-CV BA | Test BA | Training − test |
|---|---|---|---|---|---|
| FBCSP all | block | 0.999 | 0.342 | 0.353 | +0.647 |
| GA single-obj. | block | 0.751 | 0.568 | 0.353 | +0.398 |
| GA-NSGA (proposed) | block | 0.576 | 0.512 | 0.345 | +0.231 |
| EEGNet (32 ch) (seed 0) | block | 0.639 | val 0.364 | 0.350 | +0.290 |
| EEGNet (8 ch) (seed 0) | block | 0.648 | val 0.373 | 0.350 | +0.298 |
| FBCSP all | cross-session | 0.997 | 0.361 | 0.337 | +0.660 |
| GA single-obj. | cross-session | 0.753 | 0.552 | 0.328 | +0.425 |
| GA-NSGA (proposed) | cross-session | 0.547 | 0.491 | 0.340 | +0.207 |
| EEGNet (32 ch) (seed 0) | cross-session | 0.762 | val 0.372 | 0.330 | +0.432 |
| EEGNet (8 ch) (seed 0) | cross-session | 0.638 | val 0.407 | 0.330 | +0.307 |

Training BA = resubstitution on the outer training set (FBCSP methods) or the fit part of it (EEGNet); EEGNet's middle column is its validation BA.
