# DanceMonkey

Brain--Machine Interface coursework project for causal hand trajectory
estimation from macaque motor cortical spike trains.

## Repository Structure

    .
    ├── doc/
    ├── literature-review/
    ├── report/
    └── src/

## Pipelines

### Hard Direction Decoder

`src/hard-direction-decoder/`

### Time-Dependent PCA Decoder

`src/time-dependent-pca-decoder/`

The comparison results of different versions are in the corresponding directories.

## How to Run

In MATLAB (We use MATLAB R2025b, earlier versions are supported):

``` matlab
cd src
```

Run example:

``` matlab
testFunction_for_students("hard-direction-decoder/v1")

testFunction_for_students("time-dependent-pca-decoder/v5")
```

### Notes on Test Functions

-   `testFunction_for_students_MTb`
    -   Original competition function
    -   Does NOT measure runtime
    -   Only evaluates decoding performance
-   `testFunction_for_students`
    -   Extended version
    -   Measures:
        -   RMSE
        -   Runtime
    -   Computes final weighted score(0.9*RMSE+0.1*Runtime)

## Report

See `report/BrainMachineInterface.pdf` for full details.

## Repository and Group Members

https://github.com/shengj1ang/DanceMonkey

Yunxiang Cai, Sheng Jiang, Minghan Li, Haoqi Zhang
Department of Bioengineering, Imperial College London
Email: {y.cai25, sheng.jiang25, minghan.li25, haoqi.zhang25}@imperial.ac.uk
