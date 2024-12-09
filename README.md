# CS594-ANN-Project
" Implementation of Geometric Data Structures "


Overview:

This repository contains the code implementation for the CS594 project, focusing on Approximate Nearest Neighbor (ANN) queries using quadtree data structures. The project involves:

- Constructing a static quadtree for ANN queries.
- Extending the quadtree to handle (1 + ε)-approximation for varying values of ε.
- Implementing dynamic operations such as deletions and insertions.
- Conducting extensive experiments to analyze query performance under different conditions.


Project Features:

Static Quadtree:

- Constructs a quadtree from a dataset (PQuad.txt).
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

Static Quadtree: Construction time, tree height, and spread analysis.

Approximation Analysis: Trade-offs between ε values and query accuracy.

Dynamic Operations: Impact of deletions and insertions on query performance.

Sample figures and metrics are available in the results/figures folder.


References:

Erickson, J. Static to Dynamic Data Structures.

Arya, S., Mount, D. M., et al. (1998). An Optimal Algorithm for Approximate Nearest Neighbor Searching in Fixed Dimensions.

Samet, H. (2006). Foundations of Multidimensional and Metric Data Structures.


Contributors:

Anuraag Reddy Kommareddy (UIN – 668011844)

Arthi Aneel (UIN – 668779590)

Abhishikth Pammi (UIN – 674258235)


