# Randomized Algorithms 


This repository contains the Python implementation of assignments for the **Randomized Algorithms (RA-MIRI)** course.


## Assignment 1 - Galton Board

**File:**  
`Assignment1/galton_board_simulation.py`

**Description:**  
Simulation of a Galton board and comparison of empirical ball landing distribution with theoretical binomial and normal distributions.

**Requirements:**  
- Python 3.8+
- numpy  
- matplotlib

Install:

```bash
pip install numpy matplotlib
```

Run:

```bash
python Assignment1/galton_board_simulation.py
```


## Assignment 2 — Balanced Allocation

**Files & Folders:**

Assignment2/  
  ├─ ballanced-alloc.py  
  ├─ balanced_alloc_data/        # generated CSV files  
  └─ balanced_alloc_graphs/      # generated plots  


**Description:**  
Experiments for the balanced allocation problem (including d-choice strategies). Produces CSV data and graphs.

**Requirements:**  
Python 3.8+
-numpy
-pandas
-matplotlib
-tqdm

Install:
```bash
pip install numpy pandas matplotlib tqdm
```



## Assignment 3 — Cardinality Estimation
**Files & Folders:**


Assignment3/  
  ├─ cardinality-estimation.py  
  └─ results_summary.csv         # generated results  
  
**Description:** Implementation and experimental analysis of cardinality estimation algorithms for data streams.  
Algorithms: HyperLogLog (HLL), Recordinality (REC), and K-Minimum Values (KMV).  
Experiments: Tested on real datasets (Project Gutenberg novels) and synthetic Zipfian data streams.  
Output: Generates results_summary.csv with estimation errors and plots comparing theoretical vs. empirical performance (e.g., memory impact on "Dracula" text).  

**Requirements:**

Python 3.8+
-numpy
-mmh3
-matplotlib
-tqdm

Install:  
```bash  

pip install numpy mmh3 matplotlib tqdm

python Assignment3/cardinality-estimation.py
```
