# Job Shop and Flow Shop Scheduling Web Application

This Django web application provides an interface for job shop and flow shop scheduling problems with various constraints and optimization criteria.

## Features

### Flow Shop Scheduling
- Multiple dispatching rules:
 - SPT (Shortest Processing Time)
 - LPT (Longest Processing Time) 
 - FIFO (First In First Out)
 - LIFO (Last In First Out)
 - EDD (Earliest Due Date)
 - CDS (Campbell Dudek Smith)
 - MILP (Mixed Integer Linear Programming)
 - GA (Genetic Algorithm)

- Supported constraints:
 - None (Basic flow shop)
 - SDST (Sequence-Dependent Setup Times)
 - No-idle
 - No-wait
 - Blocking

### Job Shop Scheduling
- Scheduling methods:
 - Jackson's Algorithm
 - MILP Optimization

### Performance Metrics
The application calculates several key performance metrics:
- Makespan (Cmax): Maximum completion time across all jobs
- Total Flow Time (TFT): Sum of completion times
- Mean Flow Time (TT): Average completion time
- Maximum Tardiness (EMAX): Maximum lateness value
- Total Tardiness (TFR): Sum of all tardiness values
- Mean Tardiness (TAR): Average tardiness

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Windows OS

## Installation

1. **Create and activate a virtual environment**

```bash
.venv/Scripts/activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```
  

3. **Run the server**
```bash
python manage.py runserver
```

4. **Open the browser and go to the following URL**
```bash
http://localhost:8000/
```