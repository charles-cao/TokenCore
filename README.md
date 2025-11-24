# TokenCore

The datasets are in data file, Demo.ipynb shows the key steps of TextCore.

## Quick Start

### Basic Usage

```python
from TextCore import TextCore

model = TextCore()

# Fit the model
model.fit(X_train)

# Get anomaly scores
scores = model.decision_function(X_test)
```

Demo.ipynb shows the key steps of TextCore.