from typing import Tuple

import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
from matplotlib.widgets import Button, Slider
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from pathlib import Path

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.b_data_profile import get_numeric_columns
from assignments.assignment1.e_experimentation import process_life_expectancy_dataset, \
    process_amazon_video_game_dataset

from assignments.assignment3.b_simple_usages import plotly_map, plotly_tree_map, plotly_table


'''
!!!NOTE ON TESTING!!!
I have commented everything but the function I coded for in the main function, 
but feel free to play around with them as you see fit.
'''


##############################################
# Now let's use dash, a library built on top of flask (a backend framework for python) and plotly
# Check the documentation at https://dash.plotly.com/
# For a complete example, check https://dash-bootstrap-components.opensource.faculty.ai/examples/iris/
# Example(s). Read the comments in the following method(s)
##############################################


def dash_simple_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    The focus is to create a fig from plotly and add it to dash, but differently from using just plotly, now we can use html elements,
    such as H1 for headers, Div for dividers, and all interations (buttons, sliders, etc).
    Check dash documentation for html and core components.
    """
    app = dash.Dash(__name__)

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    # You create a fig just as you did in a_
    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        dcc.Graph(
            id='example-graph',
            figure=fig  # and include the fig here as a dcc.Graph
        )
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_with_bootstrap_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    See examples of components from the bootstrap library at https://dash-bootstrap-components.opensource.faculty.ai/docs/components/alert/
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig1 = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    fig2 = px.line(df, x="Fruit", y="Amount", color="City")

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.Div(children='Dash: A web application framework for Python.'), md=4),
            dbc.Col(dbc.Button('Example Button', color='primary', style={'margin-bottom': '1em'}, block=True), md=8)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph1', figure=fig1)),
            dbc.Col(dcc.Graph(id='example-graph2', figure=fig2))
        ])
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_callback_example():
    """
    Here is a more complex example that uses callbacks. With this example, I believe you will suddenly perceive why dash (and webapps)
    are so much better for visual analysis.
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        html.Hr(),
        dbc.FormGroup([
            dbc.Label("Choose Dataset"),
            dcc.Dropdown(id="dropdown", value=1,
                         options=[{"label": "First Data", "value": 1}, {"label": "Second Data", "value": 2}]),
        ]),
        dbc.FormGroup([
            dbc.Label(id='slider-value'),
            dcc.Slider(id="slider", min=1, max=10, step=0.5, value=1),
        ]),
        dbc.Button('Run Callback', id='example-button', color='primary', style={'margin-bottom': '1em'}, block=True),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph')),
            # Not including fig here because it will be generated with the callback
        ])
    ])

    @app.callback(  # See documentation or tutorial to see how to use this
        Output('example-graph', 'figure'),
        # Outputs is what you wish to update with the callback, which in this case is the figure
        [Input('example-button', 'n_clicks')],
        # Use inputs to define when this callback is called, and read from the values in the inputs as parameters in the method
        [State('dropdown', 'value'),
         # Use states to read values from the interface, but values only in states will not trigger the callback when changed
         State('slider',
               'value')])  # For example, here if you change the slider, this method will not be called, it will only be called when you click the button
    def update_figure(n_clicks, dropdown_value, slider_value):
        df2 = df[:]
        df2.Amount = df2.Amount * slider_value
        if dropdown_value == 1:
            return px.bar(df2, x="Fruit", y="Amount", color="City", barmode="group")
        else:
            return px.line(df2, x="City", y="Amount", color="Fruit")

    @app.callback(Output('slider-value', 'children'), [Input('slider', 'value')])
    def update_slider_value(slider):
        return f'Multiplier: {slider}'

    #  You can also use app.callback to get selection from any of the plotly graphs, including tables and maps, and update anything you wish.
    #  See some examples at https://dash-gallery.plotly.host/Portal/

    return app


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################

def dash_task():
    """
    There is only only one task to do, a web app with:
    1. Some nice title
    2. One visualization placeholder for dataset visualization
        a. A dropdown to allow me to select which dataset I want to see (iris, video_game and life_expectancy)
        b. Two other dropdowns for me to choose what column to put in x and what column to put in y of the visualization
        c. Another dropdown for me to choose what type of graph I want (see examples in file a_) (at least 3 choices of graphs)
        d. Feel free to change the structure of the dataset if you prefer (e.g. change life_expectancy so that
            there is one column of "year", one for "country" and one for "value")
    4. A https://dash-bootstrap-components.opensource.faculty.ai/docs/components/card/ with the number of rows being showed on the above graph
    5. Another visualization with:
        a. It will containing the figure created in the tasks in a_, b_ or c_ related to plotly's figures
        b. Add a dropdown for me to choose among 3 (or more if you wish) different graphs from a_, b_ or c_ (choose the ones you like)
        c. In this visualization, if I select data in the visualization, update some text in the page (can be a new bootstrap card with text inside)
            with the number of values selected. (see https://dash.plotly.com/interactive-graphing for examples)
    """

    '''
    My Dash app is coded below!
    The drop downs allow to choose datasets, x's and y's to be displayed for a Line, Bar, and Scatter plot.
    The number of rows are then displayed in a card.
    The next section allows for the selection of a Map, Treemap, or Data Table from file b_.
    The type of visualization is then displayed in another card.
    
    The loading times are very dependent on the preprocessing of the data, so that holds everything up a little bit.
    It also seems that the map, treemap, and data table all take a little while to load so you'll have to wait around 7
    seconds after selecting a new visualization. 
    '''

    '''
    The App
    '''
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    '''
    The Data
    I also round up all of the column names here for the drop downs 
    '''
    iris_data = read_dataset(Path('..', '..', 'iris.csv'))
    cols = get_numeric_columns(iris_data)
    iris_data = iris_data[cols]

    iris_choices = []
    for col in iris_data.columns:
        iris_choices.append({'label': col, 'value': col})

    video_game_data = process_amazon_video_game_dataset()
    # drop duplicates for efficiency reasons
    video_game_data = video_game_data.drop_duplicates(subset=['asin'])
    cols = get_numeric_columns(video_game_data)
    video_game_data = video_game_data[cols]

    video_game_choices = []
    for col in video_game_data.columns:
        video_game_choices.append({'label': col, 'value': col})

    expect_data = process_life_expectancy_dataset()
    cols = get_numeric_columns(expect_data)
    expect_data = expect_data[cols]

    expect_choices = []
    for col in expect_data.columns:
        expect_choices.append({'label': col, 'value': col})

    '''
    The HTML 
    '''

    app.layout = dbc.Container([
        # LAY OUT FOR PART 1
        html.H1(children='What a nice title'),
        html.Div(children='The following are the containers required for this task.'),
        html.Hr(),
        dbc.FormGroup([
            dbc.Label('Choose Dataset'),
            dcc.Dropdown(id='choose_dataset',
                         value='iris',
                         options=[{'label': 'Iris', 'value': 'iris'},
                                  {'label': 'Video Game', 'value': 'video_game'},
                                  {'label': 'Expectancy', 'value': 'expect'}
                                  ],
                         clearable=False,
                         searchable=False),
        ]),

        dbc.FormGroup([
            dbc.Label('Choose X'),
            dcc.Dropdown(id='choose_x', value='sepal_length',
                         options=[{'label': 'Hold', 'value': 1}],
                         clearable=False,
                         searchable=False),
        ]),

        dbc.FormGroup([
            dbc.Label('Choose Y'),
            dcc.Dropdown(id='choose_y',
                         value='sepal_width',
                         options=[{'label': 'Hold', 'value': 1}],
                         clearable=False,
                         searchable=False),
        ]),

        dbc.FormGroup([
            dbc.Label('Choose Graph'),
            dcc.Dropdown(id='choose_graph',
                         value='scatter',
                         options=[{'label': 'Line', 'value': 'line'},
                                  {'label': 'Bar', 'value': 'bar'},
                                  {'label': 'Scatter', 'value': 'scatter'}
                                  ],
                         clearable=False,
                         searchable=False),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='first_graph')),
        ]),
        # LAYOUT FOR PART 2
        html.Div([
            dbc.Card(
                dbc.CardBody([
                    html.H4("Number of Rows"),
                    html.P(
                        id='number_of_rows',
                        children={}
                    ),
                ]),
            ),
        ]),

        # LAYOUT FOR PART 3
        dbc.FormGroup([
            dbc.Label('Choose Figure'),
            dcc.Dropdown(id='choose_second_graph',
                         value='map',
                         options=[{'label': 'Plotly Map from b_', 'value': 'map'},
                                  {'label': 'Plotly Treemap from b_', 'value': 'treemap'},
                                  {'label': 'Plotly Table from b_', 'value': 'table'}
                                  ],
                         clearable=False,
                         searchable=False),
        ]),

        dbc.Row([
            dbc.Col(dcc.Graph(id='second_graph')),
        ]),

        html.Div([
            dbc.Card(
                dbc.CardBody([
                    html.H4("Number of Values Selected"),
                    html.P(
                        id='number_of_values',
                        children={}
                    ),
                ]),
            ),
        ]),
    ])
    '''
    The Callback Functions
    '''
    '''
    First Callback Function
    Depending on the chosen data set, the x and y selection drop downs will fill up with columns from that data set.
    '''
    @app.callback(
        [Output(component_id='choose_x', component_property='options'),
         Output(component_id='choose_y', component_property='options'),
         Output(component_id='choose_x', component_property='value'),
         Output(component_id='choose_y', component_property='value')
         ],
        [Input(component_id='choose_dataset', component_property='value')]
    )
    def x_y(dataset_choice):

        if dataset_choice == 'iris':
            x = iris_choices
            y = iris_choices
            val_x = iris_choices[0]['value']
            val_y = iris_choices[1]['value']
            return x, y, val_x, val_y
        elif dataset_choice == 'video_game':
            x = video_game_choices
            y = video_game_choices
            val_x = video_game_choices[0]['value']
            val_y = video_game_choices[1]['value']
            return x, y, val_x, val_y
        elif dataset_choice == 'expect':
            x = expect_choices
            y = expect_choices
            val_x = expect_choices[0]['value']
            val_y = expect_choices[1]['value']
            return x, y, val_x, val_y
    '''
    Second Callback Function
    Outputs the figure when you select the x and y values as well as the graph type
    Will also update the card showing the number of rows
    '''
    @app.callback(
        [Output(component_id='first_graph', component_property='figure'),
         Output(component_id='number_of_rows', component_property='children')],
        [Input(component_id='choose_dataset', component_property='value'),
         Input(component_id='choose_x', component_property='value'),
         Input(component_id='choose_y', component_property='value'),
         Input(component_id='choose_graph', component_property='value')]
    )
    def update_graph(dataset_choice, x_choice, y_choice, graph_choice):
        if dataset_choice == 'iris':
            if graph_choice == 'bar':
                return px.bar(iris_data, x=x_choice, y=y_choice), 'Rows: ' + str(iris_data.shape[0])
            elif graph_choice == 'line':
                return px.line(iris_data, x=x_choice, y=y_choice), 'Rows: ' + str(iris_data.shape[0])
            elif graph_choice == 'scatter':
                return px.scatter(iris_data, x=x_choice, y=y_choice), 'Rows: ' + str(iris_data.shape[0])
        elif dataset_choice == 'video_game':
            if graph_choice == 'bar':
                return px.bar(video_game_data, x=x_choice, y=y_choice), 'Rows: ' + str(video_game_data.shape[0])
            elif graph_choice == 'line':
                return px.line(video_game_data, x=x_choice, y=y_choice), 'Rows: ' + str(video_game_data.shape[0])
            elif graph_choice == 'scatter':
                return px.scatter(video_game_data, x=x_choice, y=y_choice), 'Rows: ' + str(video_game_data.shape[0])
        elif dataset_choice == 'expect':
            if graph_choice == 'bar':
                return px.bar(expect_data, x=x_choice, y=y_choice), 'Rows: ' + str(expect_data.shape[0])
            elif graph_choice == 'line':
                return px.line(expect_data, x=x_choice, y=y_choice), 'Rows: ' + str(expect_data.shape[0])
            elif graph_choice == 'scatter':
                return px.scatter(expect_data, x=x_choice, y=y_choice), 'Rows: ' + str(expect_data.shape[0])
        else:
            return px.bar(iris_data, x='sepal_length', y='sepal_width'), 'Rows: ' + str(iris_data.shape[0])
    '''
    Third Callback Function
    Calls the function from file b_ depending on the selected visualization and outputs it as a figure
    Will also update the card showing which type of visualization was selected
    '''
    @app.callback([Output(component_id='second_graph', component_property='figure'),
                   Output(component_id='number_of_values', component_property='children')],
                  Input(component_id='choose_second_graph', component_property='value')
                  )
    def update_second(choose_graph):
        if choose_graph == 'map':
            return plotly_map(), 'Map from b_ using life_expectancy data set.'
        elif choose_graph == 'treemap':
            return plotly_tree_map(), 'Tree map from b_ using life_expectancy data set.'
        elif choose_graph == 'table':
            return plotly_table(), 'Data Table from b_ using life_expectancy data set.'

    return app


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    app_se = dash_simple_example()
    app_b = dash_with_bootstrap_example()
    app_ce = dash_callback_example()
    app_t = dash_task()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # app_se.run_server(debug=True)
    # app_b.run_server(debug=True)
    # app_ce.run_server(debug=True)
    # app_t.run_server(debug=True)
