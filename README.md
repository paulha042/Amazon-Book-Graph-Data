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
A100 High Ram (80 VRAM). Therefore, I limit the scale to use the existing graph provided by
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

