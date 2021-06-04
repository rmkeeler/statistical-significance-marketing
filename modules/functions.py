import plotly.graph_objects as go
import os
from PIL import Image

def create_dashboard(figs, filename):
    """
    Takes a list of plotly figures and creates from them an HTML document. The document
    displays the charts in list order down its length.
    """
    with open(filename, 'w') as f:
        f.write('<html><head></head><body>' + '\n')
        for fig in figs:
            innerhtml = fig.to_html().split('<body>')[1].split('</body>')[0]
            f.write(innerhtml)
        f.write('</body></html>' + '\n')

def save_images(figs, save_path):
    """
    Takes a list of plotly figures and saves them to save_path as .webp files.

    Webp is pro-web format. That's why it's used, here.
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for fig in figs:
        filename = fig.layout.title.text.lower().replace(' ','_')
        file = save_path + '/' + filename + '.webp'

        fig.write_image(file)

        im = Image.open(file)
        im.show()
