from InquirerPy import prompt
from rich.console import Console
from rich.table import Table
import pyfiglet, click
import pandas as pd
import numpy as np
import kmeans_model as kmeans

def askInputInfo(option_list): 
    questions = [
        {"type": "input", "message": f'{option_list[0]}:'}, 
        {"type": "input", "message": f'{option_list[1]}:'}, 
        {"type": "input", "message": f'{option_list[2]}:'}, 
        {"type": "input", "message": f'{option_list[3]}:'}, 
        {"type": "input", "message": f'{option_list[4]}:'}, 
        {"type": "input", "message": f'{option_list[5]}:'}, 
        {"type": "input", "message": f'{option_list[6]}:'}, 
        {"type": "input", "message": f'{option_list[7]}:'}, 
        {"type": "input", "message": f'{option_list[8]}:'}, 
        {"type": "input", "message": f'{option_list[9]}:'}
    ]
    return prompt(questions)

def askPreference(): 
    return prompt([
        {"type":"confirm", "message": "Do you wish to evaluate these features upon preference?"}
    ])

def askUniversity(): 
    return prompt([
        {"type": "input", "message": f'Enter your dream university:'}
    ])

@click.command
def main(): 
    '''
    YourCollege CLI script for retreiving output from the trained KMeans Clusters
    '''
 
    print(pyfiglet.figlet_format("YourCollege CLI", font = "slant" ))

    console = Console()
    console.print("Hello there everyone!", style="bold red")
    console.print("This is a CLI application which uses certain preferences \nto help you find colleges similar to the your dream college. \n\n", style="cyan")
    
    table = Table(title="List of Features")
    table.add_column("Feature List", justify="left", style="cyan", no_wrap=True)
    feature_list = ["Type of University (Private or Public)",
            "Total undergrad enrollment",
            "College Expenditure per Student",
            "Percent of student body in STEM",
            "Women",
            "Acceptance Rate",
            "Percentage of incoming students in Top 10% of HS",
            "Graduation Rate",
            "Diversity",
            "SAT Score"]
    for feature in feature_list: 
        table.add_row(feature)
    console.print(table)
    print() 

    ranked_features = []
    feature_weights = []

    if askPreference()[0]: 
        console.print("\n--Evaluate the following features out of 1-9--", style="bold red")
        ranked_features = askInputInfo(feature_list)

        
        for i in range(len(ranked_features)):  
            feature_weights += ranked_features[i]

        table2 = Table()
        table2.add_column("Feature List", justify="left", style="cyan", no_wrap=True)
        table2.add_column("Weights", justify="left", style="cyan", no_wrap=True)
        
        for i in range(len(feature_list)-1): 
            table2.add_row(feature_list[i], feature_weights[i])
        console.print(table2)


    # Trained college list
    data = None
    if feature_weights == []: 
        data = kmeans.train()
    else: 
        data = kmeans.train(feature_weights)
    names = data.iloc[:,0].values.tolist()

    table3 = Table(title="List of Universities") 
    table3.add_column("Names", justify="left", style="cyan", no_wrap=True)
    table3.add_column("Names", justify="left", style="cyan", no_wrap=True)
    table3.add_column("Names", justify="left", style="cyan", no_wrap=True)
    table3.add_column("Names", justify="left", style="cyan", no_wrap=True)

    names1 = names[0:64]
    names2 = names[65:128]
    names3 = names[129:193]
    names4 = names[194:258]
    names5 = names[259:323]

    for i in range(63):
        table3.add_row(names1[i], names2[i], names3[i], names4[i],names5[i]) 

    console.print(table3)

    console.print("\n--FINALLY--", style="bold red")
    university_name = askUniversity()
    

    similar_colleges = kmeans.find_colleges(data, university_name)

    table4 = Table(title="List of Similar Colleges")
    table4.add_column("Names", justify="left", style="cyan", no_wrap=True)

    for i in range(similar_colleges.shape[0]): 
        table4.add_row(similar_colleges.iloc[i,0])
    
    console.print(table4)

if __name__ == "__main__": 
    main() 



