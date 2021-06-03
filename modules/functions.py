import plotly.graph_objects as go

def create_dashboard(figs, filename):
    """
    Takes a list of figures and creates from them an HTML document. The document
    displays the charts in list order down its length.
    """
    with open(filename, 'w') as f:
        f.write('<html><head></head><body>' + '\n')
        for fig in figs:
            innerhtml = fig.to_html().split('<body>')[1].split('</body>')[0]
            f.write(innerhtml)
        f.write('</body></html>' + '\n')
