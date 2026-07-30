# Day 2: Multi-Feature Regression

## What I learned
- Real problems often have multiple inputs, not just one (e.g. size, rooms, age)
- Formula becomes: y_pred = w1*x1 + w2*x2 + w3*x3 + b
- Instead of writing each term by hand, we store weights in one array and
  use np.dot(X, w) to compute the weighted sum automatically — works for
  any number of features, not just 3
- Everything else (loss, gradient descent) works exactly like Day 1 —
  the only difference is w is now an array instead of a single number

## What confused me
- (write a line here, even something small)

## Result
- Target was w=[3, 5, -2], b=10
- My model found: w=___, b=___ (fill in from your actual output)

## Practice Task
- Same concept, different feature data
- Target was: w=___, b=___
- My model found: w=___, b=___