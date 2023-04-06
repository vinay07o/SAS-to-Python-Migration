"""Pytest Cases"""
import matplotlib.pyplot as plt
import statsmodels.api as sm
from stats import DataVisualizer

                    
def test_create_scatterplot(sample_data):
    dv = DataVisualizer(sample_data)
    dv.create_scatterplot(xvar="Book_Sales", yvar="Music_Sales", groupvar="Gender", plottype="sgplot")
    assert True

def test_create_histogram(sample_data):
    dv = DataVisualizer(sample_data)
    dv.create_histogram(var='Music_Sales')
    fig_num = plt.gcf().number  # get the figure number of the current figure
    assert plt.fignum_exists(fig_num)

def test_create_probplot(sample_data):
    dv = DataVisualizer(sample_data)
    dv.create_probplot(var='Book_Sales')
    fig_num = plt.gcf().number  
    assert plt.fignum_exists(fig_num)

def test_create_boxplot(sample_data):
    dv = DataVisualizer(sample_data)
    dv.create_boxplot(xvar="Region", yvar="Book_Sales")
    fig_num = plt.gcf().number  
    assert plt.fignum_exists(fig_num)

def test_create_correlation_matrix(sample_data):
    dv = DataVisualizer(sample_data)
    dv.create_correlation_matrix()
    fig_num = plt.gcf().number  
    assert plt.fignum_exists(fig_num)


def test_create_linear_regression(sample_data):
    vis = DataVisualizer(sample_data)
    vis.create_linear_regression('Total_Sales', ['Advertising', 'Book_Sales', 'Music_Sales'])
    # Assert that the output of the model summary is not empty
    assert len(sm.OLS(sample_data['Total_Sales'], sm.add_constant(sample_data[['Advertising', 'Book_Sales', 'Music_Sales']])).fit().summary().as_text()) > 0
