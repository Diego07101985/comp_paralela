from vector_space_similarity import ServiceTextMining
from models import PdfOutput, DocumentSimilary
from classification.lazy import Knn
from kmeans_spark import K_MeansSpark
from kmeans_sequencial import K_MeansSeq

import numpy as np


import csv
import string
import glob2


def main():
    files = []

    # Utilizar este para testes longa  1000 docs

    fileNames = sorted(glob2.glob("./20_newsgroups/**/[0-9]*"))

    # Utilizar este para testes rapidos 11 docs
    # fileNames = sorted(glob2.glob("./longo/*.txt"))

    for filename in fileNames:
        files.append(open(filename, "r+", encoding="ISO-8859-1").read())

    print(len(fileNames))

    serviceTextMining = ServiceTextMining()
    terms = serviceTextMining.select_terms(files)
    matriz_tf = serviceTextMining.create_matriz_itf_terms(terms)
    matriz_df = serviceTextMining.create_matriz_idf_terms(terms, files)
    matriz_tf_df = serviceTextMining.create_matriz_tf_df_terms(
        matriz_tf, matriz_df)

    # kmeansq = K_MeansSeq(3)
    kmeansp = K_MeansSpark(3)

    # print("K means seq", kmeansq.execute(matriz_tf_df))
    print("K means Spark", kmeansp.execute(matriz_tf_df))


if __name__ == "__main__":
    main()
