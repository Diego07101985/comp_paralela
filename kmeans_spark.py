
import numpy as np
from matplotlib import style
import math
import random
from scipy import spatial
from scipy.spatial import distance
from metrics import Metrics
from pyspark.sql import SparkSession
from scipy.sparse import csc_matrix, SparseEfficiencyWarning


style.use('ggplot')


class K_Means:
    def __init__(self, k):
        self.k_clusters = k

    # Calcule os centróides para os clusters, calculando a média de todos os pontos de dados que pertencem a cada cluster.

    # Returns the sum and the count of all the documents in each cluster
    def get_cluster_stats(self, document_1_tuple, document_2_tuple):
        document_1 = document_1_tuple[0]
        document_2 = document_2_tuple[0]

        document_1_vector = document_1[1]
        document_2_vector = document_2[1]

        document_1_count = document_1_tuple[1]
        document_2_count = document_2_tuple[1]

        document_sum = [sum(x)
                        for x in zip(document_1_vector, document_2_vector)]
        tot_count = document_1_count + document_2_count

        return ((0, document_sum), tot_count)
    # Retona o index documento com o valor mas proximo do x`

    def closest_centroid(self, document, centroids):
        metrics = Metrics()
        closest = 10000000
        best = 0
        temp = 0
        var = 0
        for centroid in centroids:
            temp = metrics.get_cosine_distance(centroid[1], document[1])
            if temp != 0:
                if temp < closest:
                    closest = temp
                    best = centroid[0]
            else:
                var += 1

        if var > 2:
            return "erro"

        return best

    # # Returns the centroid cluster from the sum and count of documents in the cluster

    def get_new_clusters(self, cluster_stat):
        cluster_idx = cluster_stat[0]
        cluster_sum = cluster_stat[1][0][1]
        cluster_documents_count = cluster_stat[1][1]
        cluster_centroid = np.array(cluster_sum)
        cluster_centroid /= cluster_documents_count
        return (cluster_idx, cluster_centroid)

    def execute(self, matriz_t_idf):

        metrics = Metrics()

        document_vectors_list = []

        spark = SparkSession.builder.appName("PythonKMeans").getOrCreate()
        sc = spark.sparkContext
        document_id = 0
        cluster_stats = []

        for line in range(len(matriz_t_idf)):
            document_vector = []
            for column in matriz_t_idf[line]:
                document_vector.append(column)

            document_vectors_list.append((document_id, document_vector))
            document_id += 1

        document_vectors_rdd = sc.parallelize(document_vectors_list).cache()
        document_vectors_rdd.sortByKey()
        initial_centroids = document_vectors_rdd.repartition(
            1).takeSample(False, 3)

        # print("Centroid {0}".format(initial_centroids))

        for centroids in initial_centroids:
            print("Centroids {0}".format(centroids))
        print(initial_centroids)
        temp_dist = 1.0

        # print("Collect {0}".format(document_vectors_rdd.collect()))
        while temp_dist > 0.2:

            closest = document_vectors_rdd.map(lambda d: (
                self.closest_centroid(d, initial_centroids), (d, 1)))

            # print("Closest {0}".format(closest.collect()))

            closest = closest.filter(lambda d: d[0] != 'erro')

            # print("Closest {0}".format(closest.collect()))
            cluster_stats = closest.reduceByKey(
                self.get_cluster_stats)

            print("Closest Stats {0}".format(closest.collect()))

            new_clusters = cluster_stats.map(self.get_new_clusters).collect()

            results = []
            for index_cluster in range(len(new_clusters)):
                print(initial_centroids[index_cluster][1])
                print(new_clusters[index_cluster][1])
                results.append(metrics.get_eculedian_distance_2(
                    initial_centroids[index_cluster][1], new_clusters[index_cluster][1]))

            temp_dist = sum(results)

            print("Temp Dist {0}".format(temp_dist))
            for iK in range(len(new_clusters)):
                initial_centroids[iK] = (
                    new_clusters[iK][0], new_clusters[iK][1])

            # print(initial_centroids)

        return cluster_stats


class DocumentKmean:
    def __init__(self, id, distance_cosine, document):
        self.distance_cosine = distance_cosine
        self.document = document
        self.id = id
