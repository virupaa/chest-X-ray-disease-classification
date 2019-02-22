# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import shutil,os

dataframe = pd.read_csv("Data_Entry_2017.csv")

xy=pd.concat([pd.Series(row['Image Index'], row['Finding Labels'].split('|'))              
                    for _, row in dataframe.iterrows()]).reset_index()
     
xy.columns=['Disease','Image']


im='F:/dataset/images_vishal/'
co=-1
count=0

for i in xy['Disease']:
    co=co+1
    if i !=  "Effusion":
        count = count + 1
        print(xy['Image'][co],i)
        f = im + xy['Image'][co]
        d='F:\Eff'
        if count == 4700:
            break
        try:
            shutil.copy(f,d)  
        except FileNotFoundError:
            continue