class ReportGenerator:
    def __init__(self, analysis_results):
        self.analysis_results = analysis_results

    def generate_html_report(self, filename):
        with open(filename, 'w') as file:
            file.write(self._create_html_content())

    def _create_html_content(self):
        html_content = "<html><head><title>ASX Bank Trading Analysis Report</title></head><body>"
        html_content += "<h1>Analysis Report</h1>"
        html_content += "<h2>Results</h2>"
        html_content += "<ul>"
        for result in self.analysis_results:
            html_content += f"<li>{result}</li>"
        html_content += "</ul>"
        html_content += "</body></html>"
        return html_content

    def generate_pdf_report(self, filename):
        # Placeholder for PDF report generation logic
        pass

    def save_report(self, report_type='html', filename='report.html'):
        if report_type == 'html':
            self.generate_html_report(filename)
        elif report_type == 'pdf':
            self.generate_pdf_report(filename)
        else:
            raise ValueError("Unsupported report type. Use 'html' or 'pdf'.")