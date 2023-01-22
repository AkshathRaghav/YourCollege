from InquirerPy import prompt
from rich.console import Console
from rich.table import Table
import pyfiglet, click, math
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


def learnMore(options_list): 
    questions = [{
        "type": "list",
        "message": "Enter the university you want to learn more about:",
        "choices": options_list,
    }]
    
    return prompt(questions)

def askContinue(): 
    return prompt([
        {"type": "confirm", "message": f'Do you wanna continue?'}
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

        # print(ranked_features)
        
        for key in ranked_features.keys():
            feature_weights += [int(ranked_features[key])]
        

        print(feature_weights)

        table2 = Table()
        table2.add_column("Feature List", justify="left", style="cyan", no_wrap=True)
        table2.add_column("Weights", justify="left", style="cyan", no_wrap=True)
        
        print(type(feature_weights[0]))
        for i in range(len(feature_list)): 
            table2.add_row(feature_list[i], str(feature_weights[i]))

        print(feature_weights)
        console.print(table2)


    # Trained college list
    data = []
    feature_weights = []
    if feature_weights == []: 
        data = kmeans.train([1,1,1,1,1,1,1,1,1,1])
    else: 
        data = kmeans.train(feature_weights)
    names = data.iloc[:,0].values.tolist()

    print(data)

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
    university_name = askUniversity()[0]
    
    similar_colleges = kmeans.find_colleges(data, university_name)

    print()
    table4 = Table(title="List of Similar Colleges")
    table4.add_column("Names", justify="left", style="cyan", no_wrap=True)
    

    for i in range(similar_colleges.shape[0]): 
        table4.add_row(similar_colleges.iloc[i,0])
    
    console.print(table4)


    while askContinue()[0]:
        name = learnMore(similar_colleges.iloc[:,0].values.tolist())[0]
        find_more = similar_colleges.loc[similar_colleges['NAME'] == name]
        table5 = Table(title="Information")
        table5.add_column('Feature', justify="left", style="cyan", no_wrap=True)
        table5.add_column(f'{name}', justify="left", style="cyan", no_wrap=True)

        table5.add_row("Total Number of Undergrads:", str(find_more.iloc[0]["F.Undergrad"]))
        priv_or_pub = "Private"
        if find_more.iloc[0]["Private"] == 0: 
            priv_or_pub = "Public"
        table5.add_row("College Type:", priv_or_pub)
        table5.add_row("Average SAT Score:", str(round(find_more.iloc[0]["SAT%"]*1600)))
        table5.add_row("Percent of students in STEM:", str(math.trunc(find_more.iloc[0]["stem_percent"]*100))+"%")
        table5.add_row("Acceptance Rate:", str(math.trunc(find_more.iloc[0]["acceptance_rate"]*100))+"%")
        table5.add_row("Acceptance Rate:", str(math.trunc(find_more.iloc[0]["grad_rate"]*100))+"%")

        console.print(table5)

    console.print("\n-------------------------------- \nThanks for using me <3", style="bold red")
if __name__ == "__main__": 
    main() 
