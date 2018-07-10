*Supervised Local Modeling for Interpretability*

Model interpretability is an increasingly important component of practical machine learning. Some of the most common forms of interpretability systems are example-based, local, and global explanations. One of the main challenges in interpretability is designing explanation systems that can capture aspects of each of these explanation types, in order to develop a more thorough understanding of the model. We address this challenge in a novel model called SLIM that uses local linear modeling techniques along with a dual interpretation of random forests (both as a supervised neighborhood approach and as a feature selection method). SLIM has two fundamental advantages over existing interpretability systems. First, while it is effective as a black-box explanation system, SLIM itself is a highly accurate predictive model that provides faithful self explanations, and thus sidesteps the typical accuracy-interpretability trade-off. Second, SLIM provides both example- based and local explanations and can detect global patterns, which allows it to diagnose limitations in its local explanations.



This repo contains an implementation of SLIM as well as the code to replicate the results in the [paper](https://arxiv.org/abs/1807.02910).
