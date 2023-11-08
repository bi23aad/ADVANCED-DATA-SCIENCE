# Applied Data Science 1
"""  
Assignment 1: Visualization:Task is to apply three types if visualization 
method(graghts) to extract meaning information
"""

# Question 1

# importing packages
import pandas as pd
import matplotlib.pyplot as plt

def terrorism_fatality(eu_terrorism_fatalities_by_country):
    """ 
    define a function to produce a line plot showing multiple lines
    with proper labels and legend.
    """
    # Read the csv file
    terrorism_by_country = pd.read_csv("eu_terrorism_fatalities_by_country.csv")
    print(terrorism_by_country)
    
    # Assigning variables
    x = terrorism_by_country['year']
    y = terrorism_by_country['United Kingdom']
    y1 = terrorism_by_country['Spain']
    y2 = terrorism_by_country['Italy']
    y3 = terrorism_by_country['Greece']
    
    # set figure size
    plt.figure(figsize=(8, 6))
    
    # plotting Line Plot
    plt.figure()
    plt.plot(x, y, '-', label="United Kingdom")
    plt.plot(x, y1, '-,', label="Spain")
    plt.plot(x, y2, ':', label="Italy")
    plt.plot(x, y3, '--', label="Greece")
    
    # add legend and title
    plt.legend()
    plt.title("Terrorism Fatality in some Countries")
    
    # display plot and save
    plt.savefig("line plot.png")
    plt.show()
    return()

# Using the function
terrorism_fatality("eu_terrorism_fatalities_by_country.csv")

# Question 2

def Murder_case(murder_2015_final):
    """ 
    define a function to produce a bar chart for murder cases in the United State
    in the year 2014.
    """
    # read csv file
    Murder_Case = pd.read_csv("murder_2015_final.csv")
    print(Murder_Case)
    
    # Data to plot on bar chat
    Number_state = Murder_Case[['state', '2014_murders']]
    
    # creating bar chart
    plt.figure(figsize=(12, 6))  # figure size
    plt.bar(Number_state['state'], Number_state['2014_murders'])
    plt.xlabel("state")
    plt.ylabel("Number of Murders (2014)")
    plt.title("Number of Murder Cases by State in 2014")
    
    # for better readibility
    plt.xticks(rotation=90, fontsize=10, fontstyle='italic', fontweight='bold')  
    
    plt.tight_layout()  # to prevent clipping labels
    
    # display plot and save
    plt.savefig("Barchat.png")
    plt.show()
    return()

# Using the function
Murder_case("murder_2015_final.csv")


def Murder_case(murder_2015_final):
    """
    define a function to product a pie chart for the murders that occured in 
    selected states in US in the year 2015
    """
    # read csv file
    Murder_Case = pd.read_csv("murder_2015_final.csv")
    print(Murder_Case)
    
    #using pie chart to analyse 2015 muder distributiion across  6 states
    Murder_Case = Murder_Case.groupby('state')['2015_murders'].sum().reset_index()
    
    # selecting six preferred state
    Murder_dist = Murder_Case.loc[0:5]
    
    # create pie chart
    plt.figure()
    plt.pie(Murder_dist["2015_murders"], labels=Murder_dist["state"],\
            autopct='%d%%', startangle=140)
    plt.title("Murder case distribution 2015")
    plt.axis("equal")
    
    # display plot and save
    plt.savefig("Piechart.png")
    plt.show()
    return()

# Using the function
Murder_case("murder_2015_final.csv")
