import json
import plotly
from modules.classes import BinomialExperiment

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Scatter

app = Flask(__name__)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    labels = [x for x in df.columns if x not in ['id','message','original','genre']]
    label_means = [df[x].mean() for x in labels]

    top_vals, top_labels = sort_lists(label_means, labels, 'descending', 6)
    bot_vals, bot_labels = sort_lists(label_means, labels, 'ascending',6)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Scatter(
                    x = x,
                    y = y_null,
                    mode = 'lines'
                ),
                Scatter(
                    x = x,
                    y = y_alt,
                    mode = 'lines'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {'data':[
            Scatter(
                x = top_labels,
                y = top_vals,
                mode = 'lines'
            )
        ],

        'layout':{
            'title':'Top 6 Categories Represented in Message Set',
            'yaxis':{
                'title': '% of Messages Classified'
            },
            'xaxis': {
                'title': 'Alert Category'
            }
            }
        },

        {'data':[
            Scatter(
                x = bot_labels,
                y = bot_vals,
                mode = 'lines'
            )
        ],
        'layout':{
            'title':'Bottom 6 Categories Represented in Message Set',
            'yaxis':{
                'title':'% of Messages Classified'
            },
            'xaxis':{
                'title': 'Alert Category'
            }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()
