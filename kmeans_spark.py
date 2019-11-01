
import numpy as np
from matplotlib import style
import math
import random
from scipy import spatial
from scipy.spatial import distance
from metrics import Metrics
from pyspark.sql import SparkSession

style.use('ggplot')


class K_Means:
    def __init__(self, k):
        self.k_clusters = k

    # Calcule os centróides para os clusters, calculando a média de todos os pontos de dados que pertencem a cada cluster.
    def compute_centroids(self, lista_centroids):
        new_centroids = []

        for k, centroid in lista_centroids:

            total_distance_cossine = 0
            count_doc = 0
            distance_cossines = []
            for doc in centroid:
                if doc.distance_cosine == 0:
                    continue
                count_doc += 1
                distance_cossines.append(doc.distance_cosine)
                total_distance_cossine += doc.distance_cosine
            print(count_doc)
            mean = (total_distance_cossine / count_doc)

            print("Centroid {0}".format(len(centroid)))
            print("total_distance_cossine {0}".format(total_distance_cossine))
            print("Mean {0}".format(mean))
            print("Distances {0}".format(distance_cossines))

            distance_mean = distance_cossines.index(
                min(distance_cossines, key=lambda x: abs(x - mean)))

            new_centroids.append(
                (len(distance_cossines), total_distance_cossine))

        return new_centroids

    # Retona o index documento com o valor mas proximo do centroid
    def closest_document(self, documents, centroids):

        metric = Metrics()
        list_centroid = []
        list_closest = []
        best_document = ""
        best_index = 0

        documents_k = documents[:]

        # posso clusterizar
        for centroid in range(len(centroids)):
            list_closest = []
            lista_best_document = []
            for k, document in documents_k:
                distance_cosine = metric.get_cosine_distance(
                    centroids[centroid][1], document)

                list_closest.append(DocumentKmean(
                    k, distance_cosine, document))

            list_centroid.append((centroids[centroid][0], list_closest))

        for index_lista in range(len(list_centroid)):
            if len(lista_best_document) == 0:
                lista_best_document = [
                    doc.distance_cosine for doc in list_centroid[index_lista][1]]
            else:
                lista_best_document = [max(value) for value in np.array(list(zip([doc.distance_cosine for doc in list_centroid[index_lista][1]], [
                    distance_cosine for distance_cosine in lista_best_document])))]

        for k in range(len(list_centroid)):
            array_remove = []
            for index in range(len(lista_best_document)):
                if lista_best_document[index] > list_centroid[k][1][index].distance_cosine:
                    array_remove.append(index)
            for index in sorted(array_remove, reverse=True):
                del list_centroid[k][1][index]

        # remove o elementos que não apresentaram nenhum valor de coseno para naão enviesar a media
        for k in range(len(list_centroid)):
            for doc in list_centroid[k][1]:
                if doc.distance_cosine == 0:
                    list_centroid[k][1].remove(doc)

        # posso clusterizar

        return list_centroid

    def execute(self, matriz_t_idf):

        spark = SparkSession.builder.appName("PythonKMeans").getOrCreate()
        sc = spark.sparkContext
        cluster_stats = []

        # for line in range(len(matriz_t_idf)):
        #     document_vector = []
        #     for column in matriz_t_idf[line]:
        #         document_vector.append(column)

        #     document_vectors_list.append((document_id, document_vector))
        #     document_id += 1

        # initial_centroids = random.sample(document_vectors_list, k=3)

        # temp_dist = 1.0
        # metrics = Metrics()
        # cluster_stats = []

        # init_centroid_eu = []
        # for ik, d in initial_centroids:
        #     init_centroid_eu.append((ik, (1, 1)))

        # while temp_dist > 0.01:
        #      # Calcule os centróides para os clusters, calculando a média de todos os pontos de dados que pertencem a cada cluster.
        #     cluster_stats = self.closest_document(
        #         document_vectors_list, initial_centroids)

        #     # Returns the centroid cluster from the sum and count of documents in the cluster
        #     new_clusters = self.compute_centroids(cluster_stats)
        #     results = []

        #     # some as distancias euclidianas dos novos clusters com os iniciais esse resultado tende a zero quando não ouver mas mudanças
        #     for index_cluster in range(len(new_clusters)):
        #         results.append(metrics.get_eculedian_distance(
        #             init_centroid_eu[index_cluster][1], new_clusters[index_cluster]))

        #     temp_dist = sum(results)

        #     for ik in range(len(new_clusters)):
        #         init_centroid_eu[ik] = (0, new_clusters[ik])

        #     print("Temp = {0}".format(temp_dist))

        return cluster_stats


class DocumentKmean:
    def __init__(self, id, distance_cosine, document):
        self.distance_cosine = distance_cosine
        self.document = document
        self.id = id
