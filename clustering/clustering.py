from face_recognition.test.recognize_dataset_random import train_data
from sklearn.preprocessing import LabelEncoder
from face_recognition.clustering.functions import *
import numpy as np
import logging, pickle, time
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from face_recognition.clustering.svm_functions import train_svms

def train_cluster(n_clusters=None):
    # initializing
    start_timestamp=time.time()
    dataset=load_file(TRAIN_PATH)
    if n_clusters == None:
        n_clusters=int(0.1 * len(dataset))
    data,labels=[],[]
    dic={}
    names=[]
    k=0
    for label,feature in dataset.items():
        dic.update({k:label})
        names.append(label)
        k+=1
        data.append(feature[0])
        labels.append([label for i in range(len(feature))])

    # shuffle train dataset
    labels=np.concatenate(labels)
    data=np.concatenate(data)
    data,labels_y=pair_shuffle(data,labels)

    # create encoder
    labels = LabelEncoder().fit_transform(labels_y)

    # train process
    logger.info('Training cluster for {} labels'.format(len(labels)))
    model = KMeans(init='k-means++',
                   n_clusters=n_clusters,
                   n_init=50, tol=1e-8, max_iter=100,
                   random_state=10
                   )

    model.fit(data)
    logger.info("Model trained successfully")
    ctrs = model.cluster_centers_
    classes=make_clusters(ctrs,names,dic, dataset)

    ctr=model.cluster_centers_
    classes, ctr=cut_clusters(classes,ctr)
    n_clusters=len(classes)
    clusters = [{"classes": classes[i], "center": ctr[i],
                 "svm":{"model":None, "dic":None}} for i in range(n_clusters)]

    with open(CLUSTER_PATH, "wb") as file:
        pickle.dump(clusters, file)

    return clusters, {"message":"Clusters saved on {}".format("./clusters"),
                      "model": model, "time":time.time() - start_timestamp}

def test_model():
    logger.info("Entered on reduce test process")
    # load test dataset
    dataset=load_file(TEST_PATH)
    # load clusters vector
    SVM_AND_CLUSTERS_PATH = os.getenv("SVM_AND_CLUSTERS_PATH")
    clusters=load_file(SVM_AND_CLUSTERS_PATH)
    logger.info("Resources successfully loaded")

    centroids=[clusters[i]["center"] for i in range(len(clusters))]
    svms_metrics=[]
    print("------------------------------------------------------------------")
    for idc, cluster in enumerate(clusters):
        acc=0
        tot=0
        # logger.info("Testing classes {}".format(cluster["classes"]))
        for name in cluster["classes"]:
            data=dataset[name]
            tot+=len(data)
            for vector in data:
                pred=cluster["svm"]["model"].predict_proba(vector)[0]
                values=[]
                arg1=np.array(pred).argmax()
                values.append(pred[arg1])
                pred[arg1]=0
                arg2=np.array(pred).argmax()
                values.append(pred[arg2])
                pred[arg2]=0
                arg3=np.array(pred).argmax()
                values.append(pred[arg3])
                # logger.info("Dic for this cluster: {}".format(cluster["svm"]["dic"]))
                test=cluster["svm"]["dic"][arg1]
                test2=cluster["svm"]["dic"][arg2]
                test3 = cluster["svm"]["dic"][arg3]
                if test == name:
                    acc+=1
                elif test2 == name:
                    acc+=1
                elif test3 == name:
                    acc+=1

        print("------------------------------------------------------------------")
        logger.info("Cluster's {} classes: {}".format(
            idc, cluster["classes"]
        ))
        logger.info("Accuracy for cluster's {} svm: {}".format(
            idc, acc/tot
        ))
        svms_metrics.append(acc/tot)
        print("------------------------------------------------------------------")


    y_predict= []
    for user, vec in dataset.items():
        preds=[compute_distance(centroids, vet) for vet in vec]
        y_predict.append({"user":user, "preds":preds})

    logger.info("Features were predicted with centroids")

    acc=0
    users_acc=[]
    failed_users=[]
    for i in range(len(y_predict)):
        user=y_predict[i]["user"]
        preds=y_predict[i]["preds"]
        user_acc=0
        for index, centers in enumerate(preds):
            arg = centers[0]
            arg1 = centers[1]
            arg2 = centers[2]
            if user in clusters[arg]["classes"]:
                acc += 1
                user_acc += 1

            elif user in clusters[arg1]["classes"]:
                acc += 1
                user_acc += 1

            elif user in clusters[arg2]["classes"]:
                acc += 1
                user_acc += 1
            else:
                failed_users.append(user)
        users_acc.append(user_acc/len(preds))
    c=Counter(failed_users)
    logger.info("Mean for svms accuracy: {}".format(sum(svms_metrics)/len(svms_metrics)))
    for user, n in c.items():
        logger.info("User: {} failed on {} of {} test".format(user, n, len(dataset[user])))

    return sum(users_acc)/len(users_acc), [len(c["classes"]) for c in clusters]

def data_process():
    x, y, a, b, c = train_data(80)
    save_train_features(x)
    save_test_features(y)

def cluster_process():
    train_cluster()
    logger.info("Cluster training process concluded")
    train_svms()
    logger.info("Svms for clusters were treined")

# data_process()
# cluster_process()
acc,c=test_model()
print(acc, c)
