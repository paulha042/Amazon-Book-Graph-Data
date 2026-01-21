# Amazon-Book-Graph-Data

## My Motivation

When I was in my Bachelor degree and after studying COMP2200 Data Science, I was very
interested in building a recommendation system. However, I remember back in that time,
there was no notebooks on Kaggle showing how I should do it (until now no existing
notebooks are publicly available on Kaggle).

It was just a pre-built module, but I didn't have that much interest. This semester, when doing
this unit, I realized how the recommendation system works, they are built on graphs.
Therefore, I decided to work on Amazon Book data to build my recommendation system. On
the first days of this project, I was looking at real csv datasets, containing Amazon products
from multiple categories. Then I combined data, transformed from csv datasets to graph
data. As such, when I completed training, I can recall top k-recommendations for each user
ID.

However, since the size is too big, I can never load the data to the memory, even if I use
A100 High Ram (80 VRAM) on Google Colab. Therefore, I limit the scale to use the existing graph provided by
PyG (also known  pytorch geometric).

In this report, I implemented Graph Recommender with a Multi-Scale Attention framework
model, which I was inspired by [a research paper](https://www.nature.com/articles/s41598-025-17925-y?fbclid=IwY2xjawNz0ytleHRuA2FlbQIxMABicmlkETFGRmtwYUZsVXBiaXJ4WUFMAR7Uk7A8YCCZhKxZqsDq8U98urJzTbrLnUHUwmAKIV_20vZgPvTiQD3WNXDC2A_aem_e11BmvvjuguKIyJTsgqxxw)
. Specific details are provided later in the
report. In terms of metrics, two commonly used **Recall@K** and **NDCG@K** are implemented.

- Recall@K is used to evaluate the proportion of the K recommended items that match
the user’s actual interactions.

- NDCG@K further incorporates the ranking of the recommended items, giving higher
weights to correct recommendations that appear higher in the ranking.

- The higher the values of these two metrics, the better the quality of the model’s
recommendations. In this study, K=20 is selected to analyze the experimental results.

--- 

## 1. Environment Requirements

***a. System Components:***

- GPU: NVIDIA A100 (80 GB RAM or VRAM)
- CUDA: 12.1 (auto-matched by PyTorch 2.3+)
- cuDNN: ≥ 8.9
- Python: 3.10 or 3.11

***b. Installation Commands***

- !pip install torch
- !pip install torch-geometric
- !pip install numpy
- !pip install matplotlib
- !pip install tqdm
- !pip install networkx

Since the graph data is extremely heavy, when loaded into the model it may cause out of
memory (OOM). If you would like to run safely, you can modify some components such as
number of layers, dim (dimensions) ... The performance might be different, but it is the only
choice we can do.

## 2. Amazon Book Graph Data

In the AmazonBook dataset, the original data structure is heterogeneous, consisting of two
distinct node types:

- User nodes representing customers, and
- Book nodes representing items.

Edges describe interactions (e.g., user rates book), meaning the data is stored in a **bipartite
form.**

While this format captures semantic relationships, it is not **directly compatible** with most
Graph Neural Network (GNN) architectures, such as **GCN, GAT, or GraphSAGE**, which are
typically designed for homogeneous graphs where all nodes share a unified feature space.

Since the graph is too big, we will visualize some nodes to have an imagination of what the data looks like.

<img width="592" height="579" alt="Image" src="https://github.com/user-attachments/assets/c84c51b6-7571-4344-8734-76d4dddf2a16" />

Visually, the graph looks sparse, they are not connected to each other. Admittedly, it is a random selection, but we can confirm this by printing a line how density a graph is.

<img width="300" height="200" alt="Image" src="https://github.com/user-attachments/assets/02c70789-d4de-4390-8ff9-77f81a967717" />

### 2.1. Graph Sparsity

Looking at the graph density, less than **0.03% of all possible edges** exist, and over **91,000
nodes** are completely disconnected. Such extreme sparsity indicates that the majority of
users interact with only a few items, while a small subset of popular books receive many
connections.

### 2.2. Limitations of the Heterogeneous Graph

While this bipartite representation accurately captures user–item relationships, it introduces
two key limitations for Graph Neural Networks (GNNs):

1. Limited Message Passing: Information can only flow between users and items, never
within the same type.
This restricts feature propagation and weakens node representation learning.
2. Disconnected Components: With so many isolated or weakly linked nodes, the GNN
struggles to aggregate global context or learn stable embeddings.

### 2.3 Conversion to a Homogeneous Graph

To enable effective message passing and embedding propagation across both users and
items, the graph must be converted to a homogeneous representation.
This transformation re-indexes all nodes into a single continuous node space, where:

- User nodes occupy the index range [0, num_users)
- Item nodes are offset to [num_users, num_users + num_items)
  
The conversion process duplicates each edge in the reverse direction, ensuring that the
resulting graph is bidirectional.

This is crucial for GNN aggregation since information must flow both from users to items and
from items back to users.

| Heterogeneous Graph  | Homogeneous Graph |
|-----------------|----------------|
| Two disjoint node types (Users,Books)    | Unified node index space  |
| Directional edge: User → Book    | Bidirectional: User ↔ Book  |
| Incompatible with standard GNN layers   | Compatible with standard message passing  |

This conversion to homogeneous allows the model to **learn shared latent representations** across node types and supports **efficient neighborhood aggregation**.
