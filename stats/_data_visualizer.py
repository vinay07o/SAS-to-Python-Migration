import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from IPython.display import display, HTML
import matplotlib.pyplot as plt


class DataVisualizer:
    """
    A class for creating various visualizations of data using Python.

    Parameters:
    -----------
    data : pandas DataFrame
        The data to be visualized.

    Attributes:
    -----------
    data : pandas DataFrame
        The data to be visualized.

    Methods:
    --------
    create_scatterplot(xvar, yvar, groupvar=None, plottype='sgplot'):
        Creates a scatter plot.

    create_histogram(var):
        Creates a histogram of the specified variable.

    create_probplot(var):
        Creates a probability plot of the specified variable.

    create_boxplot(xvar, yvar):
        Creates a boxplot of the specified variables.

    create_correlation_matrix():
        Creates a scatter plot matrix with correlation table.

    create_linear_regression(depvar, indvars):
        Creates a linear regression model.

    """

    def __init__(self, data):
        """
        Initialize the class with data.

        Parameters:
        -----------
        data : pandas DataFrame
            The data to be visualized.
        """
        self.data = data

    def create_scatterplot(self, xvar, yvar, groupvar=None, plottype="sgplot"):
        """
        Creates a scatter plot.

        Parameters:
        -----------
        xvar : str
            The name of the x-axis variable.
        yvar : str
            The name of the y-axis variable.
        groupvar : str, optional
            The name of the variable for grouping the data, by default None.
        plottype : str, optional
            The type of plot to create ('sgplot' or 'gplot'), by default 'sgplot'.

        Returns:
        --------
        None
        """
        # Set default plot type
        if plottype == "gplot":
            plt.plot(self.data[xvar], self.data[yvar], "o")
        else:
            plt.scatter(self.data[xvar], self.data[yvar], marker=".")
        # Add group variable if specified
        if groupvar:
            groups = self.data[groupvar].unique()
            for i, group in enumerate(groups):
                group_data = self.data[self.data[groupvar] == group]
                if plottype == "gplot":
                    plt.plot(group_data[xvar], group_data[yvar], "o", label=group)
                else:
                    plt.scatter(
                        group_data[xvar], group_data[yvar], marker=".", label=group
                    )
        # Add axis labels and legend
        plt.xlabel(xvar)
        plt.ylabel(yvar)
        if groupvar:
            plt.legend()
        plt.show()

    def create_histogram(self, var):
        """
        Creates a histogram of the specified variable.

        Parameters:
        -----------
        var : str
            The name of the variable to be plotted.

        Returns:
        --------
        None
        """
        sns.histplot(data=self.data, x=var, kde=True)
        plt.show()

    def create_probplot(self, var):
        """
        Create a probability plot of a variable using scipy and matplotlib.

        Parameters:
        -----------
        var : str
            The name of the variable to be plotted.

        Returns:
        --------
        None
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        res = stats.probplot(self.data[var].dropna(), plot=ax)
        ax.set_title(f"Probability plot of {var}")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Ordered values")
        plt.show()


    def create_boxplot(self, xvar, yvar):
        """
        Create a boxplot using Seaborn.

        Parameters:
        -----------
        xvar : str
            The name of the variable to be plotted on the x-axis.
        yvar : str
            The name of the variable to be plotted on the y-axis.

        Returns:
        --------
        None
        """
        sns.boxplot(data=self.data, x=xvar, y=yvar)
        plt.show()


    def create_correlation_matrix(self):
        """
        Create a scatter plot matrix and display the correlation table.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        sns.set(style="white")
        sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
        # Create scatter plot matrix
        plot = sns.pairplot(self.data, diag_kind="hist", kind="reg")
        plot.fig.suptitle("Scatter Plot Matrix with Correlation Table", y=1.08)
        # Add correlation table
        corr = self.data.select_dtypes("int64").corr()
        corr_text = corr.style.background_gradient(
            cmap="coolwarm", vmin=-1, vmax=1, axis=None
        )
        text = corr_text.set_table_attributes('style="font-size: 10px"').set_caption(
            "Correlation Table"
        )
        # Display the graph with the correlation table
        display(text)
        plt.show()


    def create_linear_regression(self, depvar, indvars):
        """
        Create a linear regression model and return the summary.

        Parameters:
        -----------
        depvar : str
            The name of the dependent variable.
        indvars : list of str
            The names of the independent variables.

        Returns:
        --------
        str
            The summary of the linear regression model.
        """
        # Add a constant term for the intercept
        X = sm.add_constant(self.data[indvars])
        # Fit the model
        model = sm.OLS(self.data[depvar], X).fit()
        # Print the summary
        return model.summary()
