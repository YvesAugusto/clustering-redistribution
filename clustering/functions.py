from sklearn.cluster import KMeans
from numpy.linalg import norm as l2
import pickle, os, random, time
import numpy as np
from clustering-redistribution.log import logger
from collections import Counter

TRAIN_PATH = os.getenv("TRAIN_PATH")
TEST_PATH=os.getenv("TEST_PATH")
CLUSTER_PATH=os.getenv("CLUSTER_PATH")

# function to shuffle data and label vectors in correspondance to each other
def pair_shuffle(x,y):
    c=list(zip(x,y))
    random.shuffle(c)
    x,y=zip(*c)
    return np.array(x), np.array(y)

# compute the average of a list
def Average(lst):
    return sum(lst) / len(lst)

# return the arguments of three greatest elements on a list
def three_greatest(v):
    a=v
    a=sorted(a)
    return v.index(a[0]), v.index(a[1]), v.index(a[2])

# compute the distance between a feature vector and each one of the k-means' centroids
def compute_distance(centroids, vector):
    return three_greatest([l2(vector-c) for c in centroids])

# load a file given it's name
def load_file(filename):
    with open(filename, "rb") as file:
        dataset=pickle.load(file)
        #logger.info("Dataset len: {}".format(len(dataset)))
        return dataset

# return user train features given it's name
def get_features_from_user(name):
    # loading train a catalogues
    with open(TRAIN_PATH, "rb") as file:
        train=pickle.load(file)

    # separing train and test features
    # logger.info('Getting features from user {}'.format(name))
    if name in train:
        train_features=train[name]
    else:
        logger.info("User {} not enrolled on dataset".format(name))
        return None, None, None
    # logger.info('len-train: {}'.format(len(train_features)))
    # logger.info('len-test: {}'.format(len(test_features)))

    t = np.array(train[name])

    return (np.reshape(t,(t.shape[0], t.shape[2])))

def lowest(v):
    a = v
    a = sorted(a)
    return v.index(a[0])

def compute_lowest_distance(centroids, vector):
    return np.array([l2(vector - c) for c in centroids]).argmin()

# detects for an user the cluster to where it's features converges more
def reallocate_user(vets, centroids):
    classification=[]
    for v in vets:
        closest=compute_lowest_distance(centroids,v)
        classification.append(closest)
    c=Counter(classification)
    return c.most_common(1)[0][0]

# function to delete clusters filled with less than X users, and redistribute it's users
# to the remaining clusters.
def cut_clusters(classes, centers, thrs=7):
    centers=np.array(centers)
    classes=np.array(classes)
    to_cut={}
    for idc,c in enumerate(classes):
        if(len(c)<thrs):
            to_cut.update({idc:c})

    index=[index for index, users in to_cut.items()]
    new_classes = [classes[i] for i in range(len(classes)) if i not in index]
    new_centers = [centers[i] for i in range(len(centers)) if i not in index]
    logger.info("Index of wrong clusters {}".format(index))
    for idc,c in to_cut.items():
        logger.info("Cluster {} has minus than {} users".format(idc, thrs))
        # logger.info("Classes vector before remotion: {}".format(classes))
        # logger.info("Classes vector after remotion: {}".format(classes))
        for name in c:
            vets=get_features_from_user(name)
            new_cluster=reallocate_user(vets, new_centers)
            logger.info("New cluster for user {} is cluster number {}".format(name, new_cluster))
            new_classes[new_cluster].append(name)

    new_classes=np.array(new_classes)
    new_centers=np.array(new_centers)
    print("New shape for classes and centers: {}, {}".format(new_classes.shape, new_centers.shape))
    return new_classes, new_centers

# distribute the users, allong the clusters, according to which centroid
# each of it's features more tends to
def make_clusters(centers,names,dic, dataset):
    classes=[[], [], [], [], [], [], [], []]
    for idn, n in enumerate(names):
        pred_clusters=[compute_lowest_distance(centers,feat[0]) for feat in dataset[n]]
        c=Counter(pred_clusters)
        arg=c.most_common(1)[0][0]
        print(arg)
        # arg2=c.most_common(2)[1][0]
        classes[arg].append(n)
        # if c.most_common(1)[0][1] - c.most_common(2)[1][1] < 5:
        #     classes[arg2].append(n)

    return classes

#
# now the functions to the online classification process
#

def classify_in_clusters(feat):
    clusters=load_file(CLUSTER_PATH)
    centroids=[c["center"] for c in clusters]
    args = compute_distance(centroids, feat)
    return [clusters[arg] for arg in args]

def jump_to_clusters(clusters, feat):
    preds=[three_greatest(c["svm"]["model"].predict_proba(feat)) for c in clusters]
    names=[]
    for idc, c in clusters:
        names_aux=[c["svm"]["dic"][pred] for pred in preds[idc]]
        names.append(names_aux[0])
    return names




