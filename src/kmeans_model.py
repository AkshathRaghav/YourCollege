import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Trains model with provided weights. Default weights are set to 1 each.
def train(weights=[1 for i in range(0,10)]):
    data = pd.read_csv('data/3dfsMerged.csv')
    
    drop_cols = ['Top25perc','Top25perc','Outstate', 'Room.Board', 'Books','Personal','PhD', 'Terminal', 'S.F.Ratio', 'perc.alumni','rank', 'state_name', 'early_career_pay', 'mid_career_pay','make_world_better_percent','total_enrollment']
    colleges = data.drop(drop_cols, axis=1)
    colleges['acceptance_rate'] = colleges['Accept'] / colleges['Apps']
    colleges['top10'] = colleges['Top10perc'] / colleges['Enroll']
    colleges['stem_percent'] = colleges['stem_percent'] / 100
    colleges['grad_rate'] = colleges['Grad.Rate'] / 100
    colleges['diversity_score'] = 1 - colleges['White']
    for i in colleges.index:
        if colleges.iloc[i]['Private'] == 'Yes':
            colleges.loc[i, ['Private']] = 1
        else:
            colleges.loc[i, ['Private']] = 0
    
    colleges = colleges.drop(['Apps', 'Accept', 'Enroll', 'Top10perc','P.Undergrad','American Indian / Alaska Native', 'Asian', 'Black',
       'Hispanic', 'Native Hawaiian / Pacific Islander', 'White',
       'Two Or More Races', 'Unknown', 'Non-Resident Foreign',
       'Total Minority', 'Grad.Rate'], axis=1)
    colleges = colleges.dropna()
    
    scaler = MinMaxScaler(feature_range=(0,1))
    colleges[['F.Undergrad','Expend']] = scaler.fit_transform(colleges[['F.Undergrad','Expend']])
    
    
        

    
    
    
    
    
    
    
    
    
    
    