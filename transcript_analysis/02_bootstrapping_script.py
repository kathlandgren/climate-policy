import pandas as pd
import json
import numpy as np
from scipy import stats

import intercoder_reliability_functions as fun
from datetime import datetime
import matplotlib.pyplot as plt

import pickle

# Read the JSON file into a DataFrame
df = pd.read_json("input/main-at-2024-05-28-recoded-mostly.json")

#get intercoder reliability IDs list
all_inner_IDs=pd.read_csv("main_project_IDs.csv")

#select weekly IDs
weeklyIDS=all_inner_IDs.iloc[:2022]

annotator_list=df['annotator'].unique()
#annotator_list=[2,3,10,4,11,13,12,14] #for testing, exclude certain annotators
annotator_dict = {
    2: 'JE',
    3: 'EK',
    10: 'VR',
    4: 'CA',
    11: 'LU',
    13: 'PI',
    12: 'GE',
    14: 'TR',
    15: 'KA'
}
# select a subset of annotators if needed:
df = df[df['annotator'].isin(annotator_list)]
df['annotator'] = df['annotator'].map(annotator_dict)
# unpack the meta_info

expanded_columns = df['meta_info'].apply(pd.Series)
df = pd.concat([df.drop('meta_info', axis=1), expanded_columns], axis=1)

#filter the df based on weekly annotations
df = df[df['internal_id'].isin(weeklyIDS["internal_id"])]

df= pd.merge(df, all_inner_IDs[['internal_id', 'project_id']], on='internal_id', how='left')

question_labels=['climate_change','attitude','policy']
## define dataframes for each question
Q1_df = df.pivot(index='annotator', columns='internal_id', values=question_labels[0])
Q2_df = df.pivot(index='annotator', columns='project_id', values=question_labels[1])
Q3_df = df.pivot(index='annotator', columns='project_id', values=question_labels[2])

# Q1_df = Q1_df.reindex(df['annotator'].unique()).fillna(value='No answer')


Q1_choices=["Acknowledges","Neutral","Denies","Debate","Unclear"]
Q2_choices=["Expresses climate concern","Neutral","Expresses opposition to climate concern","Debate","Unclear"]
Q3_choices=["Supports","Neutral","Opposes","Debate","Unclear","Does not mention"]

coincidence_matrix_Q1=fun.compute_coincidence_matrix(Q1_choices,Q1_df)
coincidence_matrix_Q2=fun.compute_coincidence_matrix(Q2_choices,Q2_df)
coincidence_matrix_Q3=fun.compute_coincidence_matrix(Q3_choices,Q3_df)
Q1_alpha=fun.alpha(coincidence_matrix_Q1)
Q2_alpha=fun.alpha(coincidence_matrix_Q2)
Q3_alpha=fun.alpha(coincidence_matrix_Q3)

num_samples=10000
alpha_samples=np.zeros((num_samples,3))


for i in range(num_samples):

    alpha_samples[i,0]=fun.alpha(fun.bootstrap_transcripts_and_compute_coincidence_matrix(Q1_choices, Q1_df))

    alpha_samples[i,1]=fun.alpha(fun.bootstrap_transcripts_and_compute_coincidence_matrix(Q2_choices, Q2_df))

    alpha_samples[i,2]=fun.alpha(fun.bootstrap_transcripts_and_compute_coincidence_matrix(Q3_choices, Q3_df))

# Save the array to a pickle file
with open('output/alpha_samples.pkl', 'wb') as f:
    pickle.dump(alpha_samples, f)
