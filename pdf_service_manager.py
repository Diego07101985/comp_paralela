from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from urllib.request import urlopen
from pdfminer.pdfpage import PDFPage
from models import PdfOutput, HtmlPdf


class ServiceManagerPdf:
    def readPDF(self, arquivo_pdf):
        resource = PDFResourceManager()
        buffer = StringIO()
        codec = "utf-8"
        layoutParams = LAParams()
        device = TextConverter(resource, buffer, codec=codec, laparams=layoutParams)

        maxpages = 0
        caching = True
        pagenos = set()
        interpreter = PDFPageInterpreter(resource, device)

        for page in PDFPage.get_pages(
            arquivo_pdf,
            pagenos,
            maxpages=maxpages,
            caching=caching,
            check_extractable=True,
        ):
            interpreter.process_page(page)

        arquivo_pdf.close()
        device.close()
        content = buffer.getvalue()

        buffer.close()
        return content

    def setInitialConfPdf(self, pdf):
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        epw = pdf.w - pdf.l_margin - pdf.r_margin
        eph = pdf.h - pdf.t_margin - pdf.b_margin

        pdf.rect(pdf.l_margin, pdf.t_margin, w=epw, h=eph)
        pdf.set_font("Times", "", 12)
        th = pdf.font_size

        pdf.ln(th)

        pdf.ln(2 * th)
        pdf.multi_cell(epw, th, txt="")

        return pdf

    def writePdf(self, pdfOutput):

        new_pdf = HtmlPdf(orientation="P", unit="mm", format="A4")

        pdf = self.setInitialConfPdf(new_pdf)

        pdf.set_font("Arial", size=18)
        pdf.cell(200, 10, txt="Trabalho 3", ln=1, align="L")

        pdf.set_font("Arial", size=15)
        pdf.cell(200, 10, txt="1 - Matrix TF * IDF", ln=1, align="L")
        pdf.cell(
            200,
            10,
            txt="Total de Termos : " + str(len(pdfOutput.terms)),
            ln=1,
            align="L",
        )

        pdf.set_font("Arial", size=16)
        pdf.cell(
            100,
            8,
            txt="Tabela de frequência de terms em um documento: TF ",
            ln=1,
            align="L",
        )
        # Trabalho 1 - TF * IDF
        # Create table TF
        table_tf = self.create_table_tf(pdfOutput)
        pdf.set_font("Arial", size=8)

        col_width = pdf.w / 1.5
        row_height = pdf.font_size

        pdf.write_html(table_tf)

        # Create table DF

        table_df = self.create_table_df(pdfOutput)

        pdf.set_font("Arial", size=16)
        pdf.cell(
            100,
            6,
            txt="Tabela de frequência de terms entre documentos: DF ",
            ln=1,
            align="L",
        )

        col_width = pdf.w / 3.5
        row_height = pdf.font_size

        pdf.write_html(table_df)

        pdf = self.setInitialConfPdf(new_pdf)
        # Create table TF * IDF
        table_tf_df = self.create_table_tf_df(pdfOutput)
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 6, txt="TF * IDF", ln=1, align="L")

        col_width = pdf.w / 1.5
        row_height = pdf.font_size

        pdf.write_html(table_tf_df)

        # Trabalho 2 - 10 arquivos mais relevantes

        pdf = self.setInitialConfPdf(new_pdf)
        table_nmax = self.create_table_nmax(pdfOutput)

        pdf.set_font("Arial", size=18)
        pdf.cell(200, 16, txt="10 documentos mais relevantes", ln=1, align="C")
        pdf.cell(
            200, 6, txt="Tabela com os 10 documentos mais relevantes ", ln=1, align="c"
        )
        pdf.cell(200, 8, txt=" Query = " + pdfOutput.query, ln=1, align="L")
        col_width = pdf.w / 3.5
        row_height = pdf.font_size

        pdf.write_html(table_nmax)

        pdf.output("trabalho3.pdf")

    def create_table_tf(self, pdfOutput):
        table = """<table border="0" class="print-friendly" style=" border: 1px solid #000; width: 575">
        <thead><tr>
        """
        for i in range(len(pdfOutput.fileNames)):
            table += (
                "<th width='15%'>"
                + pdfOutput.fileNames[i].replace("curto/", "").replace(".txt", "")
                + "</th>"
            )
        table += "</tr></thead><tbody>"

        for line in range(len(pdfOutput.matriz_tf)):
            table += "<tr  style='margin-right:1px;margin-left:0px' >"
            table += "<td align='center' width='8%'>" + pdfOutput.terms[line] + "</td>"
            for colum in pdfOutput.matriz_tf[line]:
                table += "<td align='center' width='12%'>" + str(colum) + "</td>"
            table += "</tr>"

        table += "</tbody></table>"

        return table

    def create_table_df(self, pdfOutput):
        table_df = """<table border="0" style="margin-right:auto;margin-left:0px" width="90%">
        <thead><tr>
            <th style='margin-right:auto;margin-left:0px' width='12%'>Terms</th>
            <th style='margin-right:auto;margin-left:0px' width='12%'>DF</th>
            <th style='margin-right:auto;margin-left:0px' width='12%'>IDF</th>
        """
        table_df += "</tr></thead><tbody>"

        for line in pdfOutput.matriz_df:
            table_df += "<tr style='margin-right:auto;margin-left:0px'>"
            for colum in line:
                table_df += "<td align='center' width='12%'>" + str(colum) + "</td>"
            table_df += "</tr>"

        table_df += "</tbody></table>"

        return table_df

    def create_table_tf_df(self, pdfOutput):
        table_tf_df = """
        <table border="0" class="print-friendly" style=" border: 1px solid #000; width: 575">
        <thead><tr >
        """

        for i in range(len(pdfOutput.fileNames)):
            table_tf_df += (
                "<th width='15%'>"
                + pdfOutput.fileNames[i].replace("curto/", "").replace(".txt", "")
                + "</th>"
            )
        table_tf_df += "</tr></thead><tbody>"

        for line in range(len(pdfOutput.matriz_tf_df)):
            table_tf_df += "<tr  style='margin-right:1px;margin-left:0px' >"
            table_tf_df += (
                "<td align='center' width='8%'>" + pdfOutput.terms[line] + "</td>"
            )
            for colum in pdfOutput.matriz_tf_df[line]:
                table_tf_df += "<td align='center' width='12%'>" + str(colum) + "</td>"
            table_tf_df += "</tr>"

        table_tf_df += "</tbody></table>"

        return table_tf_df

    def create_table_nmax(self, pdfOutput):
        table_nmax = """
        <table border="0" class="print-friendly" style=" border: 1px solid #000; width: 575">
        <thead><tr>
        """
        table_nmax += (
            "<th width='30%'>Document</th><th width='30%'>Distancia dos cossenos</th>"
        )
        table_nmax += "</tr></thead><tbody>"
        for document in pdfOutput.most_relevant:
            table_nmax += (
                "<tr><td align='center'>"
                + document.file
                + "</td><td align='center'>"
                + str(document.coeficient[0])
                + "</td></tr>"
            )

        table_nmax += "</tbody></table>"
        return table_nmax
