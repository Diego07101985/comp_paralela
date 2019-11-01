import numpy as np
from nltk.tokenize import word_tokenize
from numpy import ndarray
import math
from nltk.tokenize import RegexpTokenizer
from models import DocumentSimilary, DocumentSimilaryEuclidian
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer


import glob2


class ServiceTextMining:

    _docs = []

    def strip_punctuation(self, str):
        return ''.join(c for c in str if c not in punctuation)

    def strip_number(self, str):
        return ''.join(c for c in str if not c.isdigit())

    def select_terms(self, files):
        array_terms = []
        self._docs = []
        pattern = "\w+"
        tokenizer = RegexpTokenizer(pattern)
        stop_words = stopwords.words('english')
        wordnet_lemmatizer = WordNetLemmatizer()

        filtered_sentence = []

        for i in range(len(files)):
            base_text = files[i]
            # pre processamento
            text_without_pontuaction = self.strip_punctuation(base_text)
            text_without_number = self.strip_number(text_without_pontuaction)
            word_tokens = word_tokenize(text_without_number)
            lemmatized_word = [wordnet_lemmatizer.lemmatize(
                word) for word in word_tokens]
            removing_stopwords = [
                word for word in lemmatized_word if word not in stop_words]

            if i > 0:
                self._docs.append(removing_stopwords)
                for newterm in removing_stopwords:
                    if newterm not in array_terms:
                        array_terms.append(newterm)
            else:
                self._docs.append(removing_stopwords)
                for term in set(removing_stopwords):
                    array_terms.append(term)

        return array_terms

    def create_matriz_binary_terms(self, terms):
        matriz_terms = []
        column = []

        for term in terms:
            for doc in self._docs:
                if term in doc:
                    column.append(1)
                else:
                    column.append(0)

            matriz_terms.append(column)
            column = []
        return np.transpose(matriz_terms)

    def cosine_similarity(self, matriz_terms):
        count_el_mtz = 0
        numerador = 0
        denominador = 0
        element = 0
        element2 = 0
        similaridade_cossenos = []

        while count_el_mtz < len(matriz_terms):
            first_doc = matriz_terms[count_el_mtz]

            if count_el_mtz + 1 < len(matriz_terms):
                second_doc = matriz_terms[count_el_mtz + 1]

                for indice_term in range(len(first_doc)):
                    numerador += first_doc[indice_term] * \
                        second_doc[indice_term]
                    element += math.pow(first_doc[indice_term], 2)
                    element2 += math.pow(second_doc[indice_term], 2)

                denominador = math.sqrt(element) * math.sqrt(element2)
                if numerador == 0 and denominador == 0:
                    similaridade_cossenos.append(0)
                else:
                    similaridade_cossenos.append(
                        round(numerador / denominador, 3))

            count_el_mtz += 1

        return similaridade_cossenos

    def euclidian_distance(self, matriz_terms):

        # d(p,q)=d(q,p) = raiz((q1-p1)ˆ2 +(q2-p2))
        count_el_mtz = 0
        euclidian_distance_terms = []
        calc_between_point = 0

        while count_el_mtz < len(matriz_terms):
            first_doc = matriz_terms[count_el_mtz]

            if count_el_mtz + 1 < len(matriz_terms):
                second_doc = matriz_terms[count_el_mtz + 1]

                for indice_term in range(len(first_doc)):
                    calc_between_point += math.pow(first_doc[indice_term] -
                                                   second_doc[indice_term], 2)

                euclidian_distance_terms.append(
                    round(math.sqrt(calc_between_point), 3))
            count_el_mtz += 1
        return euclidian_distance_terms

    def create_matriz_itf_terms(self, terms):
        matriz_terms = []
        column = []
        for term in terms:
            count_terms = 0
            for doc in self._docs:
                if term in doc:
                    column.append(
                        round(
                            1
                            + math.log10(
                                sum([term == doc[i] for i in range(len(doc))])
                            ),
                            3,
                        )
                    )
                else:
                    column.append(0)

            matriz_terms.append(column)
            column = []

        return np.transpose(matriz_terms)

    def create_matriz_tf_terms(self, terms):
        matriz_terms = []
        column = []

        for term in terms:
            count_terms = 0
            for doc in self._docs:
                if term in doc:
                    column.append(
                        round(sum([term == doc[i]
                                   for i in range(len(doc))])), 3
                    )
                else:
                    column.append(0)

            matriz_terms.append(column)
            column = []

        return np.transpose(matriz_terms)

    def create_matriz_idf_terms(self, terms, docs):
        matriz_terms = []
        count_term = []

        for term in terms:
            count_term_each_doc = 0
            for doc in docs:
                if term in doc:
                    count_term_each_doc += 1
            if count_term_each_doc > 0:
                matriz_terms.append(
                    [
                        term,
                        count_term_each_doc,
                        round(math.log10((len(docs) / count_term_each_doc)), 3),
                    ]
                )
            else:
                matriz_terms.append([term, 0, 0])

        column = []
        return matriz_terms

    def create_matriz_tf_df_terms(self, matriz_tf, matriz_df):
        # iterate through rows of X
        result = []
        for i in range(len(matriz_tf)):
            result.append([])

            # # iterate through columns of Y
            for j in range(len(matriz_df)):
                # iterate through rows of Y
                result[i].append(round(matriz_df[j][2] * matriz_tf[i][j], 3))
        return result

    def calc_euclidian_distance_in_query(self, query):
        files = []
        fileNames = sorted(glob2.glob("./docs/**/*.txt"))

        for filename in fileNames:
            files.append(open(filename, "r+").read())

        print("Files {0}".format(len(fileNames)))
        count = 0
        coefient_doc = []
        documents = []
        coeficient_array = []

        while count < len(fileNames):
            serviceTextMining = ServiceTextMining()
            terms = serviceTextMining.select_terms([files[count], query])
            relevation_matriz_docs = serviceTextMining.create_matriz_tf_df_terms(
                serviceTextMining.create_matriz_itf_terms(terms),
                serviceTextMining.create_matriz_idf_terms(terms, files),
            )
            eu_distance = (
                serviceTextMining.euclidian_distance(relevation_matriz_docs
                                                     ),
            )

            document = DocumentSimilaryEuclidian(
                fileNames[count], eu_distance[0], fileNames[count].split("/")[2])
            documents.append(document)
            count += 1
        print("Files {0}".format(len(documents)))
        return documents

    def calc_distance_cosseno_in_query(self, query):
        files = []
        fileNames = sorted(glob2.glob("./docs/**/*.txt"))
        for filename in fileNames:
            files.append(open(filename, "r+").read())
        count = 0
        coefient_doc = []
        documents = []
        coeficient_array = []

        while count < len(files):
            serviceTextMining = ServiceTextMining()
            terms = serviceTextMining.select_terms([files[count], query])
            relevation_matriz_docs = serviceTextMining.create_matriz_tf_df_terms(
                serviceTextMining.create_matriz_itf_terms(terms),
                serviceTextMining.create_matriz_idf_terms(terms, files),
            )
            cosine_similarity = (
                serviceTextMining.cosine_similarity(relevation_matriz_docs
                                                    ),
            )

            document = DocumentSimilary(fileNames[count], cosine_similarity[0])
            documents.append(document)
            count += 1

        return documents

    def calc_distance_cosseno_in_files(self):
        files = []
        fileNames = sorted(glob2.glob("curto/*.txt"))
        for filename in fileNames:
            files.append(open(filename, "r+").read())
        count = 0
        coefient_doc = []
        coeficient_array = []

        while count < len(files):
            serviceTextMining = ServiceTextMining()
            terms = serviceTextMining.select_terms(
                [files[count], files[count - 1]])
            relevation_matriz_docs = serviceTextMining.create_matriz_tf_df_terms(
                serviceTextMining.create_matriz_itf_terms(terms),
                serviceTextMining.create_matriz_idf_terms(terms, files),
            )
            cosine_similarity = (
                serviceTextMining.cosine_similarity(relevation_matriz_docs),
            )
            document = DocumentSimilary(
                [fileNames[count], files[count - 1]], cosine_similarity
            )
            coeficient_array.append(document.coeficient)

            coefient_doc.append(document)
            count += 1

        return coeficient_array

    def Nmaxelements(self, documents, N):
        final_list = []
        selected_document = None
        selected_names = []
        array_coeficientes = []

        for value in range(0, N):
            max = 0
            for document in documents:
                if document.coeficient[0] > max:
                    max = document.coeficient[0]
                    selected_document = document

            for index_document, document_fin in enumerate(documents):
                if document_fin.file == selected_document.file:
                    del documents[index_document]
            if selected_document.file not in selected_names:
                final_list.append(selected_document)
                selected_names.append(selected_document.file)

        return final_list
