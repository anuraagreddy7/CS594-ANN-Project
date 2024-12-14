# CS594-ANN-Project
" Implementation of Geometric Data Structures "

Usage:
* Clone the repository
* Ensure Python 3.x and required libraries (numpy, matplotlib) are installed
* Place dataset.txt in the project directory
* Run main.py to execute all experiments
* Results and graphs will be generated in results/graphs folder
  
Overview:

This repository contains the code implementation for the CS594 project, focusing on Approximate Nearest Neighbor (ANN) queries using quadtree data structures. The project involves:

- Constructing a static quadtree for ANN queries.
- Extending the quadtree to handle (1 + ε)-approximation for varying values of ε.
- Implementing dynamic operations such as deletions and insertions.
- Conducting extensive experiments to analyze query performance under different conditions.


Project Features:

Static Quadtree:

- Constructs a quadtree from a dataset (dataset.txt).
- Calculates tree height and dataset spread.

Approximate Nearest Neighbor Queries:

- Supports (1 + ε)-ANN queries.
- Performance analysis for varying ε values.

Dynamic Operations:

- Deletions: Updates the quadtree after point deletions and reconstructs as needed.
- Insertions: Extends the tree to handle incremental data additions dynamically.

Experimental Analysis:

- Query performance comparison in dense and sparse regions.
- Graphs and metrics showcasing accuracy, query time, and tree properties.


Results:

- Static Quadtree: Construction time, tree height, and spread analysis.

- Approximation Analysis: Trade-offs between ε values and query accuracy.

- Dynamic Operations: Impact of deletions and insertions on query performance.

- The outputs and graphs are available in the results/graphs folder.




Contributors:

Anuraag Reddy Kommareddy (UIN – 668011844)

Arthi Aneel (UIN – 668779590)

Abhishikth Pammi (UIN – 674258235)


