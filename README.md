# ai4cpp

This repo acts as library of minimalistic implementations of AI algorithms for C++.

## Contents:

### 1. ql4cpp
Q Learning model based on epsilon-greedy approach

### 2. sarsa
SARSA based learning model based on epsilon-greedy approach

### 3. td4cpp
Temporal Differemce learning module

## Usage
For each module, you can specify the number of parameters and their range 0 to n.

Example: 
```QL test;```
```test.init();```

```test.iterate(performance_value);```

state values are stored as array : test.state[i] is the ith state parameter value for the current iteration.
