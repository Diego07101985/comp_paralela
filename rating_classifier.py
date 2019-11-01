from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_fscore_support, f1_score
import numpy as np


class Rating:
    def __init__(self):
        self.clazz_predict = None

    def confusion_matrix(self, results, docs, classes):
        values_predicts = self.conf_values_predicts(results, docs, classes)

        # print(values_predicts[1])
        # print(values_predicts[0])
        return confusion_matrix(values_predicts[1], values_predicts[0], labels=classes)

    def conf_values_predicts(self, results, docs, classes):
        actuals = self.get_documents_actual(docs, classes).elements
        new_predicts = np.copy(actuals)

        # for predict in results:
        #     index = 0
        #     for actual in docs:
        #         if predict.file == actual:
        #             if(self.clazz_predict != predict.clasz):
        #                 new_predicts[index] = self.clazz_predict

        #         index += 1
        return [actuals, new_predicts]

    def precision_value(self, results, docs, classes):
        values_predicts = self.conf_values_predicts(results, docs, classes)
        return precision_score(values_predicts[1],
                               values_predicts[0], average='weighted')

    def recall_score(self, results, docs, classes):
        values_predicts = self.conf_values_predicts(results, docs, classes)
        return recall_score(values_predicts[1], values_predicts[0], average='weighted')

    def precision_recall_fscore_support(self, results, docs, classes):
        values_predicts = self.conf_values_predicts(results, docs, classes)
        return f1_score(values_predicts[1], values_predicts[0], average='weighted', labels=classes)

    def set_class_predict(self, clazz_predict):
        self.clazz_predict = clazz_predict

    def get_documents_actual(self, docs, classes):
        total_elements_by_class = []
        elements_in_doc = []
        for clasz in classes:
            count_elements = 0
            for doc in docs:
                if doc.split("/")[2] == clasz:
                    elements_in_doc.append(clasz)
                    count_elements += 1

            total_elements_by_class.append(count_elements)

        return ActualModel(elements_in_doc, total_elements_by_class)


class ActualModel:
    def __init__(self, elements, total_elements_by_class):
        self.elements = elements
        self.total_elements_by_class = total_elements_by_class


class PredictResultModel:
    def __init__(self, elements, predictsModel):
        self.elements = elements
        self.predictsModel = predictsModel


class PredictModel:
    def __init__(self, total_elements_by_class, clazz):
        self.total_elements_by_class = total_elements_by_class
        self.clazz = clazz
