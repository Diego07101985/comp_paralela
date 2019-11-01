from fpdf import FPDF, HTMLMixin


class PdfOutput:
    def __init__(
        self,
        distance_cosseno,
        matriz_tf,
        matriz_df,
        matriz_tf_df,
        terms,
        fileNames,
        most_relevant,
        query,
    ):
        self.distance_cosseno = distance_cosseno
        self.matriz_tf = matriz_tf
        self.matriz_df = matriz_df
        self.matriz_tf_df = matriz_tf_df
        self.terms = terms
        self.fileNames = fileNames
        self.most_relevant = most_relevant
        self.query = query


class HtmlPdf(FPDF, HTMLMixin):
    pass


class DocumentSimilary:
    def __init__(self, file, coeficient):
        self.file = file
        self.coeficient = coeficient


class DocumentSimilaryEuclidian:
    def __init__(self, file, coeficient, clasz):
        self.file = file
        self.coeficient = coeficient
        self.clasz = clasz
