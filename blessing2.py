#importing needed packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stats import skew, kurtosis, bootstrap




def import_data(filename):
    """
    Load the csv file downloaded from the worldbank database, clean 
    and transpose the data.


    Returns:
        data (DataFrame) : A Dataframe of the cleaned original worldbank, with
        years as columns.
        transposed_data (DataFrame) : A DataFrame with the countries as columns
    """

    #read csv file
    brics_data = pd.read_csv("brics2.csv")

    # Replace non-number with pd.NA
    brics_data = brics_data.replace('..', pd.NA)

    # Melt the original dataframe to have 'YEAR' as a column
    melted_brics_data = pd.melt(brics_data,
                                id_vars=['Series Name', 'Series Code',
                                         'Country Name', 'Country Code'],
                                var_name='Year',
                                value_name='Value')
    print(melted_brics_data)

    # Extract year from column names
    melted_brics_data['Year'] = melted_brics_data['Year'].str.extract(
        r'(\d{4})')
    print(melted_brics_data['Year'])

    # Convert 'Year' to numeric
    melted_brics_data['Year'] = pd.to_numeric(
        melted_brics_data['Year'], errors='coerce')

    # Pivot the melted dataframe to have individual columns
    # for each country
    transposed_brics_data = melted_brics_data.pivot_table(
        index=['Year', 'Series Name', 'Series Code'],
        columns='Country Name',
        values='Value',
        aggfunc='first')
    print(transposed_brics_data)

    # Reset index for a clean structure
    transposed_brics_data.reset_index(inplace=True)

    # Convert the country columns in the transposed data to float
    columns_to_convert = ['Brazil', 'Russian Federation',
                          'India', 'China', 'South Africa']
    transposed_brics_data[columns_to_convert] = transposed_brics_data[columns_to_convert].astype(
        float)
    transposed_brics_data['Year'] = transposed_brics_data['Year'].astype(
        'category')

    # Return both Dataframes
    return brics_data, transposed_brics_data


# Function to explore statistical properties of each indicators
def exploration(data):
    """
    Use three statistical methods to explore the data. Describe, Skewness, and
    Kurtosis.

    Args:
        data (DataFrame): The transposed DataFrame.
        countries (List): A list of countries to examine.
        series_list (List): A list of the series names of the indicators to
        examine.

    Returns:
        descriptives (DataFrame): A DataFrame of the results of the summary
        statistics of the dataframe done using the .describe() function.
        other_stats (DataFrame): A DataFrame of the results of the skewness
        and kurtosis of the distribution.
    """
    # Create a new data frame of the country and parsed series
    df = data[['Brazil', 'Russian Federation',
               'India', 'China', 'South Africa']]

    # Perform descriptive statistics of the numerical variables
    # using .describe()
    descriptives = df.describe()
    print(descriptives)

    # Use the bootstrap function from the stats.py module to
    # calculate statistical properties of the distribution
    kurtosis_list = [
        f"{np.round(kurtosis(pd.to_numeric(df[x])), 4)} +/- {np.round(0.5 * (bootstrap(pd.to_numeric(df[x]), kurtosis)[1] - bootstrap(pd.to_numeric(df[x]), kurtosis)[0]), 4)}" for x in df.columns]

    # Calculate skewness
    skewness_list = [
        f"{np.round(skew(pd.to_numeric(df[x])), 4)} +/- {np.round(0.5 * (bootstrap(pd.to_numeric(df[x]), skew)[1] - bootstrap(pd.to_numeric(df[x]), skew)[0]), 4)}" for x in df.columns]

    # Convert the lists to dataframe
    bootstrap_result = pd.DataFrame({'Country': [x for x in df.columns],
                                     'Skewness': skewness_list,
                                     'Kurtosis': kurtosis_list})
    print(bootstrap_result)

    return descriptives, bootstrap_result

# Function to compare indicators using correlation analysis


def comparison(data):
    """
    Plot correlation matrix 

    Args:
        data (DataFrame): The original DataFrame.
        countries (List): A list of countries to examine.
        series_list (List): A list of the series names of the indicators to
        examine.

    Returns:
        df (DataFrame): A Dataframe of the streamlined data that was explored.
        descriptives (DataFrame): A DataFrame of the results of the summary
        statistics of the dataframe done using the .describe() function.
    """
    # Melt the original dataframe to have 'YEAR' as a column
    melted_brics_data = pd.melt(data, id_vars=['Series Name', 'Series Code',
                                               'Country Name', 'Country Code'],
                                var_name='Year',
                                value_name='Value')

    # Extract year from column names
    melted_brics_data['Year'] = melted_brics_data['Year'].str.extract(
        r'(\d{4})')

    # Convert 'Year' to numeric
    melted_brics_data['Year'] = pd.to_numeric(
        melted_brics_data['Year'], errors='coerce')

    # Pivot the data to have indicators as columns
    pivot_brics_data = melted_brics_data.pivot_table(index=['Year', 'Country Name',
                                                            'Country Code'],
                                                     columns='Series Name',
                                                     values='Value',
                                                     aggfunc='first')

    # Reset index for a clean structure
    pivot_brics_data.reset_index(inplace=True)

    # Convert the series name to float data type
    pivot_brics_data[[
        'Arable land (% of land area)',
        'Urban population (% of total population)',
        'Renewable energy consumption (% of total final energy consumption)',
        'CO2 intensity (kg per kg of oil equivalent energy use)',
        'Total greenhouse gas emissions (kt of CO2 equivalent)']] = pivot_brics_data[[
            'Arable land (% of land area)',
            'Urban population (% of total population)',
            'Renewable energy consumption (% of total final energy consumption)',
            'CO2 intensity (kg per kg of oil equivalent energy use)',
            'Total greenhouse gas emissions (kt of CO2 equivalent)']].astype(
        float)

    pivot_brics_data['Year'] = pivot_brics_data['Year'].astype('category')

    # Plot the correlation for the indicators per country
    for country in pivot_brics_data['Country Name'].unique():

        # Filter based on selected indicator codes and countries
        df = pivot_brics_data[pivot_brics_data['Country Name'] == country]
        df = pivot_brics_data.iloc[:, 3:]

        # Perform Correlation Analysis
        # Generate and visualize the correlation matrix
        corr = df.corr().round(2)
        # Mask for the upper triangle
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        # Set figure size
        f, ax = plt.subplots(figsize=(15, 9))

        # Draw the heatmap
        sns.heatmap(corr, mask=mask, cmap='Spectral', vmin=-1, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

        plt.title(
            f"Correlation Matrix of indicators for {country}",  fontsize = 25)
        plt.tight_layout()

    return pivot_brics_data




# Function visualize the data based grouped by indicators
def visualize_by_indicator(data):
    """
    Create Visualizations for storytelling.

    Args:
        data (DataFrame): The transposed DataFrame.
        countries (List): A list of countries to examine.
        series_list (List): A list of the series names of the indicators to
        examine.

    Returns:

    """
    # Create a new data frame of the country and parsed series
    countries = ['Brazil', 'Russian Federation',
                 'India', 'China', 'South Africa']

    indicators = ['Arable land (% of land area)',
                  'Urban population (% of total population)',
                  'Renewable energy consumption (% of total final energy consumption)',
                  'CO2 intensity (kg per kg of oil equivalent energy use)',
                  'Total greenhouse gas emissions (kt of CO2 equivalent)']

    # Plot barchart
    plt.figure(figsize=(15, 9))

    # Loop through the variables and plot
    for variable in indicators:
        sns.barplot(x='Country Name', y=variable,
                    hue='Year', data=data, width=.4)
        plt.title(f'{variable}',  fontsize = 25)
        plt.ylabel(None)
        plt.tight_layout()
        plt.show()


# Function visualize the data based grouped by countries
def visualize_by_country(data):
    """
    Create Visualizations for storytelling.

    Args:
        data (DataFrame): The transposed DataFrame.
        countries (List): A list of countries to examine.
        series_list (List): A list of the series names of the indicators to
        examine.

    Returns:

    """
    # Create a new data frame of the country and parsed series
    countries = ['Brazil', 'Russian Federation',
                 'India', 'China', 'South Africa']

    indicators = ['Arable land (% of land area)',
                  'Urban population (% of total population)',
                  'Renewable energy consumption (% of total final energy consumption)',
                  'CO2 intensity (kg per kg of oil equivalent energy use)',
                  'Total greenhouse gas emissions (kt of CO2 equivalent)']

    # Lineplot
    for indicator in indicators:
        indicators_df = data[data['Series Name'] == indicator]
        sns.set(style="darkgrid")

        # Plotting
        plt.figure(figsize=(15, 9))

        # Loop through the variables and plot each one with a different style
        for i, variable in enumerate(countries):
            sns.lineplot(x='Year', y=variable, data=indicators_df,
                         label=variable)

        # Add labels and title
        plt.xlabel('Year', fontsize = 25)
        plt.ylabel('Values', fontsize = 25)
        plt.title(
            f"Trend for {indicator} over the years",  fontsize = 25)

        # Add a legend
        plt.legend()

        plt.tight_layout()


# Implementation
brics, transposed_brics = import_data("brics2")

descriptives, stats_df = exploration(transposed_brics)

selected_countries = ['Russia', 'India', 'China', 'Brazil', 'South Africa']
selected_indicators = ['Arable land (% of land area)',
                       'Urban population (% of total population)',
                       'Renewable energy consumption (% of total final energy consumption)',
                       'CO2 intensity (kg per kg of oil equivalent energy use)',
                       'Total greenhouse gas emissions (kt of CO2 equivalent)']

new_brics = comparison(brics)

visualize_by_indicator(new_brics)

visualize_by_country(transposed_brics)

# # Function to filter the data
# def filter_data(data, series_name):
#     filtered_data = data[data['Series Name'] == series_name]

#     return filtered_data
