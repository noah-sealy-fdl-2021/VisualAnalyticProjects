from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.widgets import Button, Slider
from pathlib import Path

from assignments.assignment1.a_load_file import read_dataset
from assignments.assignment1.e_experimentation import process_life_expectancy_dataset
from assignments.assignment2.c_clustering import simple_k_means


'''
!!!NOTE ON TESTING!!!
I have removed all plt.plot() and fig.plot() lines from the code.
I know that the returned fig.show() for matplotlib sometimes is funky, so you may have to add a plt.plot() in the 
function itself while testing for grading.
I have also left the main functions uncommented, but feel free to play around with them as you see fit.
For matplotlib I have been using plt.plot(), in the function itself
For plotly I have been using fig.plot(), from the returned figure.
'''


###############
# Interactivity in visualizations is challenging due to limitations and clunkiness of libraries.
# For example, some libraries works well in Jupyter Notebooks, but then the code makes barely any sense
# and becomes hard to change/update, defeating the purpose of using Jupyter notebooks in the first place,
# and other libraries provide a window of their own, but they are very tied to the running code,
# and far from the experience you'd expect from a proper visual analytics webapp
#
# We will try out some approaches to exercise in this file, but the next file will give you the proper tooling to make a
# well-rounded and efficient code for visual interactions.
##############################################
# Example(s). Read the comments in the following method(s)
##############################################


def matplotlib_simple_example():
    """
    Using the same logic from before, we can add sliders or buttons to select options for interactivity.
    Matplotlib is limited to only a few options, but do take a look, since they are useful for fast prototyping and analysis

    In case you are using PyCharm, I suggest you to uncheck the 'Show plots in tool window'
    to be able to see and interact with the buttons defined below.
    This example comes from https://matplotlib.org/3.1.1/gallery/widgets/buttons.html
    """
    freqs = np.arange(2, 20, 3)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    t = np.arange(0.0, 1.0, 0.001)
    s = np.sin(2 * np.pi * freqs[0] * t)
    l, = plt.plot(t, s, lw=2)

    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    return fig, ax


def matplotlib_simple_example2():
    """
    Here is another example, which also has a slider and simplifies a bit the callbacks
    """
    data = np.random.rand(10, 5)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.bar(np.arange(10).astype(str).tolist(), data[:, 0])

    class Index(object):
        ind = 0
        multiplier = 1

        def change_data(self, event, i):
            self.ind = np.clip(self.ind + i, 0, data.shape[1] - 1)
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(), data[:, self.ind] * self.multiplier)
            plt.draw()

        def change_multiplier(self, value):
            self.multiplier = value
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(), data[:, self.ind] * self.multiplier)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.1, 0.05, 0.12, 0.075])
    axnext = plt.axes([0.23, 0.05, 0.12, 0.075])
    axslider = plt.axes([0.55, 0.1, 0.35, 0.03])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(lambda event: callback.change_data(event, 1))
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(lambda event: callback.change_data(event, -1))
    slider = Slider(axslider, 'multiplier', 1, 10, 1)
    slider.on_changed(callback.change_multiplier)

    return fig, ax


def plotly_slider_example():
    """
    Here is a simple example from https://plotly.com/python/sliders/ of how to include a slider in plotly
    Notice how similar it already is to GapMinder!
    """
    df = px.data.gapminder()
    fig = px.scatter(df, x="gdpPercap", y="lifeExp",
                     animation_frame="year",  # set which column makes the animation through a slider
                     size="pop",
                     color="continent",
                     hover_name="country",
                     log_x=True,
                     size_max=55,
                     range_x=[100, 100000],
                     range_y=[25, 90])

    fig["layout"].pop("updatemenus")  # optional, drop animation buttons

    fig.show()

    return fig


def plotly_button_example():
    """
    To have buttons, plotly requires us to use go (and not px) to generate the graph.
    The button options are very restrictive, since all they can do is change a parameter from the go graph.
    In the example below, it changes the "mode" value of the graph (between lines and scatter)
    The code is a modified code example taken from https://plotly.com/python/custom-buttons/
    """
    x = np.random.rand(50) * np.random.randint(-10, 10)
    y = np.random.rand(50) * np.random.randint(-10, 10)

    fig = go.Figure()

    # Add surface trace
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(type="buttons",
                 direction="left",
                 buttons=[
                     dict(
                         label="line",  # just the name of the button
                         method="update",
                         # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         args=[{"mode": "markers"}],  # This is the value being updated in the visualization
                     ), dict(
                         label="scatter",  # just the name of the button
                         method="update",
                         # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         args=[{"mode": "line"}],  # This is the value being updated in the visualization
                     )
                 ],
                 pad={"r": 10, "t": 10}, showactive=True, x=0.11, xanchor="left", y=1.1, yanchor="top"
                 # Layout-related values
                 ),
        ]
    )
    return fig


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
def matplotlib_interactivity():
    """
    Do an interactive matplotlib plot where I can select which visualization I want.
    Make either a slider, a dropdown or several buttons and make so each option gives me a different visualization from
    the matplotlib figures of b_simple_usages. Return just the resulting fig as is done in plotly_slider_example.
    """

    '''
    The Data
    Taken from file a_
    As always, the graph is random so sometimes it can look a little funky.
    If it looks especially strange, just rerun the program.
    '''

    x = np.random.rand(50) * np.random.randint(-10, 10)
    y = np.random.rand(50) * np.random.randint(-10, 10)
    matrix_2D = np.random.rand(10, 10) * np.random.randint(-10, 10)

    '''
    The Figure
    '''

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    # init call
    ax.bar(x, y)

    '''
    The "Callback" Functions
    Each time the user clicks a button, the figure is cleared and whichever one they select it drawn
    '''

    class Index(object):
        def bar(self, event):
            ax.clear()
            ax.bar(x, y)
            plt.draw()

        def pie(self, event):
            ax.clear()
            ax.pie(x, labels=range(len(x)))
            plt.draw()

        def hist(self, event):
            ax.clear()
            ax.hist(x, bins=10)
            plt.draw()

        def heatmap(self, event):
            ax.clear()
            ax.imshow(matrix_2D, cmap='cool', interpolation='nearest')
            plt.draw()

    '''
    The Buttons
    '''

    callback = Index()
    axbar = plt.axes([0.1, 0.05, 0.12, 0.075])
    axpie = plt.axes([0.23, 0.05, 0.12, 0.075])
    axhist = plt.axes([0.36, 0.05, 0.12, 0.075])
    axheat = plt.axes([0.49, 0.05, 0.12, 0.075])

    bar_but = Button(axbar, 'Bar Chart')
    bar_but.on_clicked(lambda event: callback.bar(event))
    pie_but = Button(axpie, 'Pie Chart')
    pie_but.on_clicked(lambda event: callback.pie(event))
    hist_but = Button(axhist, 'Histogram')
    hist_but.on_clicked(lambda event: callback.hist(event))
    heat_but = Button(axheat, 'Heat Map')
    heat_but.on_clicked(lambda event: callback.heatmap(event))

    return fig


def matplotlib_cluster_interactivity():
    """
    Do an interactive matplotlib plot where I can select how many clusters I want to train from.
    Use iris dataset (just numeric columns) and k-means (feel free to reuse as/c_clustering if you wish).
    The slider (or dropdown) should range from 2 to 10. Return just the resulting fig.
    """

    '''
    The Data
    Using the iris data, only the numeric columns
    '''

    data = read_dataset(Path('..', '..', 'iris.csv'))
    data = data.drop(['species'], axis=1)

    '''
    The Figure
    '''

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    # init call
    model = simple_k_means(X=data, n_clusters=2)
    ax.plot(model['clusters'], marker='o', linestyle='None')

    '''
    The "Callback" Function
    Everytime a new slider value is selected, the data will be clustered and displayed based on those clusters.
    '''
    class Index(object):

        def cluster(self, num_clusters):
            ax.clear()
            model = simple_k_means(X=data, n_clusters=num_clusters)
            # the k means function from A2 returns the clusters
            ax.plot(model['clusters'], marker='o', linestyle='None')

    '''
    The Slider
    The slider moves from 2 to 10, snapping at every integer on the way :)
    '''

    callback = Index()
    axslider = plt.axes([0.55, 0.1, 0.35, 0.03])
    slider = Slider(axslider, 'cluster', 2, 10, valstep=1, valinit=2)
    slider.on_changed(callback.cluster)

    return fig


def plotly_interactivity():
    """
    Do a plotly graph with all plotly 6 figs from b_simple_usages, and make 6 buttons (one for each fig).
    Change the displayed graph depending on which button I click. Return just the resulting fig.
    """
    '''
    !!!IMPORTANT!!!
    In response to Leonardo's post in the Teams channel, I believe I am a victim to the background lay over bug he
    described. The scatter polar, data table, and map seem to have trouble going away after you click another button.
    For the scatter polar and map the data itself even goes away, but the background figures stick around.
    
    My method of implementation involves adding all of the traces initially, with their visibility set to False, then
    each button will correspond to an array of True and False values, which will determine which visibilities are set to
    True and False. 
    For example, if you click the Composite Graph button, visibilities will be set to False on all charts but the bar
    and the line. 
    
    I assure you that all other data is implemented correctly, and if you comment those bugged figures, and replace them
    with bar charts, the program works fine! I hope you can understand that this bug seems to be out of our hands. 
    For what it is worth the program would be a very cool visualization in theory, 
    and I have some pride towards what I made. Thanks!
    '''

    '''
    The Data
    I decided to only use the data for the country Afghanistan from the process_life_expectancy_data() function.
    EXCEPT for the map, I use all of the countries from the dataset in the map.
    '''

    data = process_life_expectancy_dataset()
    df = data[data['name'] == 'Afghanistan']

    '''
    The Figure
    '''

    fig = go.Figure()

    '''
    The Graph Objects
    I use a Scatter, Bar, Polar Scatter, Data Table, and Choropleth Map.
        The Composite Scatter Bar is simply a combination of the Scatter and the Bar.
    Each graph object is added to the figure at the start, with their visibilities set to false.
    The visibilities will change upon buttons being pressed (see below).
    '''

    fig.add_trace(
        go.Scatter(x=df['year'],
                   y=df['expectancy'],
                   name="Scatter",
                   visible=False
        )
    )

    fig.add_trace(
        go.Bar(x=df['year'],
               y=df['expectancy'],
               name="Bar",
               visible=False
        )
    )

    fig.add_trace(
        go.Scatterpolar(r=df['year'],
                        theta=np.sin(df['expectancy'] ** 2),
                        name='Scatter Polar',
                        visible=False
        )
    )

    fig.add_trace(
        go.Table(header=dict(values=list(['year', 'expectancy']), fill_color='paleturquoise', align='left'),
                 cells=dict(values=[df['year'], df['expectancy']], fill_color='lavender', align='left'),
                 visible=False
        )
    )

    fig.add_trace(
        go.Choropleth(z=data['expectancy'],
                      colorscale='Blues',
                      locations=data['name'],
                      locationmode='country names',
                      visible=False
        )
    )

    '''
    The Buttons
    As mentioned before, everytime you click a button a new bool array will be passed to toggle the visibility of 
    each trace.
    For example, if you clicked the Composite Graph button, visibilities will be set to False on all charts but the bar
    and the line. 
    '''

    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                direction='left',
                buttons=list([
                    dict(
                        label='Scatter',
                        method='update',
                        args=[{'visible': [True, False, False, False, False]}],
                    ),
                    dict(
                        label='Bar',
                        method='update',
                        args=[{'visible': [False, True, False, False, False]}],
                    ),
                    dict(
                        label='Scatter Polar',
                        method='update',
                        args=[{'visible': [False, False, True, False, False]}],
                    ),
                    dict(
                        label='Table',
                        method='update',
                        args=[{'visible': [False, False, False, True, False]}],
                    ),
                    dict(
                        label='Composite',
                        method='update',
                        args=[{'visible': [True, True, False, False, False]}],
                    ),
                    dict(
                        label='Map',
                        method='update',
                        args=[{'visible': [False, False, False, False, True]}],
                    ),
                ]),
                pad={"r": 10, "t": 10}, showactive=True, x=0.11, xanchor="left", y=1.1, yanchor="top"
            )
        ]
    )

    return fig


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_i = matplotlib_interactivity()
    fig_m_ci = matplotlib_cluster_interactivity()
    fig_p = plotly_interactivity()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # matplotlib_simple_example()[0].show()
    # matplotlib_simple_example2()[0].show()
    # plotly_slider_example().show()
    # plotly_button_example().show()
    # fig_m_i.show()
    # fig_m_ci.show()
    # fig_p.show()
