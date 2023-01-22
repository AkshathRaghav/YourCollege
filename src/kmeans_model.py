import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Trains model with provided weights. Default weights are set to 1 each.
# Returns pandas dataframe of the colleges with clusters
def train(weights=[1 for i in range(0,10)]):
    data = pd.read_csv('../data/FINALDATA.csv')
    
    drop_cols = ['SAT25', 'SAT75', 'Top25perc','Top25perc','Outstate', 'Room.Board', 'Books',
        'Personal','PhD', 'Terminal', 'S.F.Ratio', 'perc.alumni','rank', 'state_name', 'early_career_pay', 'mid_career_pay',
        'make_world_better_percent','total_enrollment']    
    colleges = data.drop(drop_cols, axis=1)
    colleges['acceptance_rate'] = colleges['Accept'] / colleges['Apps']
    colleges['top10'] = colleges['Top10perc'] / colleges['Enroll']
    colleges['stem_percent'] = colleges['stem_percent'] / 100
    colleges['grad_rate'] = colleges['Grad.Rate'] / 100
    colleges['diversity_score'] = 1 - colleges['White']
    colleges['SAT%'] = colleges['SATAVG'] / 1600
    
    for i in colleges.index:
        if colleges.iloc[i]['Private'] == 'Yes':
            colleges.loc[i, ['Private']] = 1
        else:
            colleges.loc[i, ['Private']] = 0
    
    colleges = colleges.drop(['SATAVG','Apps', 'Accept', 'Enroll', 'Top10perc','P.Undergrad','American Indian / Alaska Native', 'Asian', 'Black',
       'Hispanic', 'Native Hawaiian / Pacific Islander', 'White',
       'Two Or More Races', 'Unknown', 'Non-Resident Foreign',
       'Total Minority', 'Grad.Rate'], axis=1)
    colleges = colleges.dropna()
    
    features = ['Private', 'F.Undergrad', 'Expend', 'stem_percent', 'Women',
       'acceptance_rate', 'top10', 'grad_rate', 'diversity_score', 'SAT%']
    
    colleges.to_csv('../predictions/college_recs.csv', index=False)
   
    for i in range(0,10):
        colleges[features[i]] = weights[i] * colleges[features[i]]
         
    scaler = MinMaxScaler(feature_range=(0,1))
    colleges[['F.Undergrad','Expend']] = scaler.fit_transform(colleges[['F.Undergrad','Expend']])
    
    kmeans = KMeans(n_clusters=40, init='k-means++', random_state=42)
    kmeans.fit(colleges.drop(['NAME'], axis=1))
    colleges['kmeans_cluster'] = kmeans.labels_  
        
    return colleges

# Takes dataframe of colleges and name of college as arguments and returns a dataframe of similar colleges
# Returns pandas dataframe of similar colleges
def find_colleges(colleges, name):
    data = pd.read_csv('../predictions/college_recs.csv')
    cluster = colleges.loc[colleges['NAME'] == name,['kmeans_cluster']].iloc[0][0]
    output = colleges.loc[colleges['kmeans_cluster'] == cluster]
    [i] = colleges.loc[colleges['NAME'] == name].index
    output = output.drop(axis=0, index=i)
    names = output.NAME.tolist()
    
    return data.loc[data['NAME'].isin(names)]
    
    
        

    
    
    
    
    
    
    
    
    
    
    