# HumanAndMachine

## Abstract
Entity resolution (ER) has wide applications and receives considerable attentions in literature.  Similarity functions can be used to judge whether two data objects refer to the same real world entity. However, the results obtained from pairwise comparison often failed to meet the transitivity, for the fact that $A \sim B$ and $B \sim C$ do not imply $A \sim C$. We devise a framework that identify those records referring to same entities by graph partitioning. By checking those edges outside the cliques, we can validate the results with minimum verification task. The experimental evaluations show that our framework outperform existing state-of-arts.

## Introduction

## Approach
There are two steps for our approach:
- Using **max_weight_matching(G)**, we obtain the optimal threshold $\theta$.
- Using the **transitivity**, to connect those matched pairs into connected components.

## Algorithm
1. construct the graph **G**, with similarity $\theta$. **max_weight_matching(G)**.
2. verify the matching with lowest similarity, e.g., $v_i$ and $v_j$.
	1. If $v_i =  v_j$, then $\theta \gets sim(v_i, v_j)$. and merge those nodes into one nodes.
	2. Else, using binary search to find the lowest $\theta^*$ such that all the matchings with $sim \geq \theta$ can be merged.
	3. If none of the matching can be merged, then stop the iteration.

Each iteration take $O(log(n))$ operations.
The total complexity can be restricted to less than $O(log^2(n))$.

## Experimental Evaluation
