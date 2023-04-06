import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
from IPython.display import display, HTML
import statsmodels.api as sm


class DataVisualizer:
    """Data Viasualization in Python. (converted from SAS)"""
    
    def __init__(self, data):
        """Initialization of all the required arguments."""
        self.data = data

    def create_scatterplot(self, xvar, yvar, groupvar=None, plottype="sgplot"):
        """Function to create a scatter plot."""
        
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
                    plt.scatter(group_data[xvar], group_data[yvar], marker=".", label=group)

        # Add axis labels and legend
        plt.xlabel(xvar)
        plt.ylabel(yvar)
        if groupvar:
            plt.legend()
        plt.show()

    def create_histogram(self, var):
        '''This function creates histogram of provided variable'''
        sns.histplot(data=self.data, x=var, kde=True)
        plt.show()

    def create_probplot(self, var):
        """Produce a probability plot using scipy and matplotlib"""
        fig, ax = plt.subplots(figsize=(8, 6))
        res = stats.probplot(self.data[var].dropna(), plot=ax)
        ax.set_title(f"Probability plot of {var}")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Ordered values")
        plt.show()
        
    def create_boxplot(self, xvar, yvar):
        """Produce a boxplot using Seaborn"""
        sns.boxplot(data=self.data, x=xvar, y=yvar)
        plt.show()
    
    def create_correlation_matrix(self):
        '''This function creates scatter plot and define co-relation with the other data'''
        sns.set(style="white")
        sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
        # Create scatter plot matrix
        plot = sns.pairplot(self.data, diag_kind='hist', kind='reg')
        plot.fig.suptitle('Scatter Plot Matrix with Correlation Table', y=1.08)
        # Add correlation table
        corr = self.data.select_dtypes('int64').corr()
        corr_text = corr.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1, axis=None)
        text = corr_text.set_table_attributes('style="font-size: 10px"').set_caption('Correlation Table')
        # Display the graph with the correlation table
        display(text)
        plt.show()

    def create_linear_regression(self, depvar, indvars):
        '''This function create linear regression model statistically'''
        
        # Add a constant term for the intercept
        X = sm.add_constant(self.data[indvars])

        # Fit the model
        model = sm.OLS(self.data[depvar], X).fit()

        # Print the summary
        return model.summary()
