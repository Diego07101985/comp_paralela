
import numpy as np
from matplotlib import style
import math
import random
from scipy import spatial
from scipy.spatial import distance
from metrics import Metrics
from pyspark.sql import SparkSession
from scipy.sparse import csc_matrix, SparseEfficiencyWarning
from vector_space_similarity import ServiceTextMining
from models import PdfOutput, DocumentSimilary
import dill
import numpy as np
import functools
import collections


import csv
import string
import glob2
from mpi4py import MPI


style.use('ggplot')


# Calcule os centróides para os clusters, calculando a média de todos os pontos de dados que pertencem a cada cluster.


def get_cluster_stats(document_1_tuple, document_2_tuple):
    document_sum = 0
    tot_count = 0

    if document_1_tuple and document_2_tuple:
        document_1 = document_1_tuple[0]
        document_2 = document_2_tuple[0]
        document_sum = []

        # print(document_2_tuple[0])
        # print(document_1_tuple[0])

        # print(document_1_tuple[0] == document_2_tuple[0])
        if document_1_tuple[0] == document_2_tuple[0]:
            document_1_vector = document_2_tuple[1][0][1]
            document_2_vector = document_2_tuple[1][0][1]

            # np.array(document_1_tuple)
            document_1_count = document_1_tuple[1][1]
            document_2_count = document_2_tuple[1][1]

            document_sum = [sum(x)
                            for x in zip(document_1_vector, document_2_vector)]
            tot_count = document_1_count + document_2_count

    return ((0, document_sum), tot_count)

    # Retona o index documento com o valor mas proximo do x`


def closest_centroid(document, centroids, comm):
    metrics = Metrics()
    closest = 10000000
    best = 0
    temp = 0
    var = 1
    for centroid in centroids:
        temp = metrics.get_cosine_distance(
            centroid[1], document[1])
        if temp != 0:
            if temp < closest:
                closest = temp
                best = centroid[0]
        else:
            var += 1
    if var > 2:
        return "erro"

    return best


def get_new_clusters(cluster_stat):
    cluster_idx = cluster_stat[0]
    cluster_sum = cluster_stat[1][1]
    cluster_documents_count = cluster_stat[2]
    cluster_centroid = np.array(cluster_sum, dtype=np.float64)
    cluster_centroid /= cluster_documents_count
    return (cluster_idx, cluster_centroid)


def execute():
    metrics = Metrics()
    document_vectors_list = []
    document_id = 0
    initial_centroids = []
    tempo_final = 0
    tempo_inicial = 0
    new_closests = []
    new_clusters_mpi = []
    temp_dist_mpi = 1
    initial_centroids_mpi = []
    document_mpi = []

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()

    if rank == 0:
        files = []
        fileNames = sorted(glob2.glob("longo/*.txt"))
        for filename in fileNames:
            files.append(open(filename, "r+").read())

        serviceTextMining = ServiceTextMining()
        terms = serviceTextMining.select_terms(files)
        matriz_tf = serviceTextMining.create_matriz_itf_terms(terms)
        matriz_df = serviceTextMining.create_matriz_idf_terms(terms, files)
        matriz_tf_df = serviceTextMining.create_matriz_tf_df_terms(
            matriz_tf, matriz_df)

        for line in range(len(matriz_tf_df)):
            document_vector = []
            for column in matriz_tf_df[line]:
                document_vector.append(column)

            document_vectors_list.append((document_id, document_vector))
            document_id += 1

        initial_centroids_mpi = random.sample(document_vectors_list, k=3)
    else:
        document_vectors_list = []
        initial_centroids_mpi = []

    comm.Barrier()

    initial_centroids_mpi = comm.bcast(initial_centroids_mpi, root=0)
    document_mpi = comm.scatter(
        document_vectors_list, root=0)

    tempo_inicial = MPI.Wtime()
    print("RUN MPI")
    while temp_dist_mpi > 0.01:

        best_centroid = {}
        reduce_closest_mpi = []
        closests = []
        reduce_closest = []
        new_clusters = []
        if rank == 0:
            num_workers = len(document_vectors_list) - 1
            closed_workers = 0

            while closed_workers < num_workers:
                status = MPI.Status()
                best = comm.recv(source=MPI.ANY_SOURCE,
                                 tag=MPI.ANY_TAG, status=status)
                if best['max_value'] != 0:
                    closests.append(
                        (best['best_index'], (best['best_vc_doc'], 1)))
                else:
                    closests.append(('erro', (best['best_vc_doc'], 1)))

                closed_workers += 1

            new_closests = [d for d in [
                d for d in closests if d[0] != 'erro'] if d[0] != []]

            for nc in new_closests:
                total_doc = 0
                document_sum = []
                closest = [d for d in reduce_closest if d[0] == nc[0]]

                if not closest:
                    reduce_closest.append(
                        (nc[0], (0, nc[1][0][1]), nc[1][1]))
                    continue

                for k, rc in enumerate(reduce_closest):
                    if rc[0] == nc[0]:
                        total_doc = closest[0][2] + nc[1][1]
                        document_sum = [sum(x)
                                        for x in zip(nc[1][0][1], closest[0][1][1])]
                        reduce_closest[k] = (
                            nc[0], (0, document_sum), total_doc)
                        break

            for rc in reduce_closest:
                new_clusters.append(get_new_clusters(rc))

        else:
            max_value = 0
            var = 0
            best = 0
            for c in initial_centroids_mpi:
                temp = metrics.get_cosine_distance(
                    c[1], document_mpi[1])
                if temp != 0:
                    if temp > max_value:
                        max_value = temp
                        best = c[0]

            best_centroid = {'best_index': best,
                             'max_value': max_value, 'best_vc_doc': document_mpi}

            comm.send(best_centroid, dest=0)

        new_clusters_mpi = comm.bcast(new_clusters, root=0)

        comm.Barrier()
        if rank == 0:
            results = []
            for index_cluster in range(len(new_clusters_mpi)):
                results.append(metrics.get_eculedian_distance(
                    initial_centroids_mpi[index_cluster][1], new_clusters_mpi[index_cluster][1]))
            temp_dist_mpi = sum(results)

        for iK in range(len(new_clusters_mpi)):
            initial_centroids_mpi[iK] = (
                new_clusters_mpi[iK][0], new_clusters_mpi[iK][1])

        initial_centroids_mpi = comm.bcast(initial_centroids_mpi, root=0)
        temp_dist_mpi = comm.bcast(temp_dist_mpi, root=0)

    if rank == 0:
        tempo_final = MPI.Wtime()
        print("Tempo ", tempo_final-tempo_inicial)
    return new_closests


class DocumentKmean:
    def __init__(self, id, distance_cosine, document):
        self.distance_cosine = distance_cosine
        self.document = document
        self.id = id


if __name__ == "__main__":
    execute()
