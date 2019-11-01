class Knn:
    def NminElements(self, documents, N):
        final_list = []
        selected_document = None
        selected_names = []
        array_coeficientes = []
        for value in range(0, N):
            min = 10000
            for document in documents:
                if min > document.coeficient[0]:
                    selected_document = document
                    min = document.coeficient[0]

            for index_document, document_fin in enumerate(documents):
                if document_fin.file == selected_document.file:
                    del documents[index_document]
            if selected_document.file not in selected_names:
                final_list.append(selected_document)
                selected_names.append(selected_document.file)
        return final_list
