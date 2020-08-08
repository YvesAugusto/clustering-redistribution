# clustering-redistribution
IA application for clustering classes, which means to classify then according to it's proximity to each other, given the number of groups.

A method for clustering classes according to the occurencies of it's samples on k-means prototypes. Then, delete those with low population, distributing it's classes along the remaining ones.

On /clustering/export-variables.sh, check where to save your train and test data. Both of them must be on the format: {"class1 name": [sample1, sample2, ..., sampleN], "class 2 name": [sample1, sample2, ..., sampleN], ...}
