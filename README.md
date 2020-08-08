# clustering-redistribution
IA application for clustering classes, which means to distribute then, given the desired number of groups.

A method for clustering classes according to the occurencies of it's samples on k-means prototypes. Then, delete those with low population, distributing it's classes along the remaining ones. It will be realized by computing, for each class, the distances between each one of it's samples and the remaining centroids, in order to vinculate the class to the cluster most times showed itself to be the nearest.

On /clustering/export-variables.sh, check where to save your train and test data. Both of them must be on the format: {"class1 name": [sample1, sample2, ..., sampleN], "class 2 name": [sample1, sample2, ..., sampleN], ...}
