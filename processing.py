# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 17:01:47 2018

@author: youko
"""

import numpy as np
import pandas as pd
from pprint import pprint
import sys
import sklearn
from sklearn import preprocessing

#OUTFILE  = open("nullvalues.txt", "w")
#sys.stdout = OUTFILE
myData = pd.read_csv('sample.csv' , sep=',', encoding='latin1')
#null_value = myData.isnull().sum()
#print(null_value.to_string())

#myData = pd.read_csv('TAMU_FINAL_DATASET_2018.csv') 

##diab row to combine the results of 'DIABETES' and 'Diab_Type'
def diab (row):
   if row['Diab_Type'] =='Diabetes Type II' :
      return 2
   if row['Diab_Type'] == 'Diabetes Type I' :
      return 1
   if row['DIABETES'] == 0 :
      return 0
   if row['Diab_Type']== 'Diabetes Unspeci':
      return 3
  
   return 'Other'
##apply the diab function to this dataframe
   ##we have not managed about blank values yet
myData['DIAB']  = myData.apply (lambda row: diab (row),axis=1)

#replace char variables to numerical. All the null value are handle with creating a new category represented by 0.
myData['SEX_CD'].replace(['F','M'],[0,1],inplace=True)
myData['ESRD_IND'].replace(['N','Y',None],[1,2,0],inplace=True)
myData['HOSPICE_IND'].replace(['N','Y',None],[1,2,0],inplace=True)
myData['PCP_ASSIGNMENT'].replace(['MEMBER SELECTED','ATTRIBUTED','UNATTRIBUTED',None],[3,2,1,0],inplace=True)
myData['DUAL'].replace(['N','Y',None],[2,1,0],inplace=True)
myData['INSTITUTIONAL'].replace(['N','Y',None],[2,1,0],inplace=True)
myData['LIS'].replace(['N','Y',None],[2,1,0],inplace=True)
myData['MCO_HLVL_PLAN_CD'].replace(['MA','MAPD',None],[2,1,0],inplace=True)
myData['MCO_PROD_TYPE_CD'].replace(['HMO','LPPO','PFFS','RPPO',None],[4,3,2,1,0],inplace=True)
myData['Dwelling_Type'].replace(['A','B','C','M','N','P','S','T', None],[8,7,6,5,4,3,2,1,0],inplace=True)
  
##drop column DIABETES and Diab_Type since there is no null values here 
del myData['DIABETES']
del myData['Diab_Type']

pprint(myData[:10])
# 
# myData['SEX_CD'].replace(['F','M'],[0,1],inplace=True)
# myData['ESRD_IND'].replace(['N','Y'],[1,2],inplace=True)
# myData['HOSPICE_IND'].replace(['N','Y'],[1,2],inplace=True)
# myData['PCP_ASSIGNMENT'].replace(['MEMBER SELECTED','ATTRIBUTED','UNATTRIBUTED',None],[3,2,1,0],inplace=True)
# myData['DUAL'].replace(['N','Y'],[2,1],inplace=True)
# myData['INSTITUTIONAL'].replace(['N','Y'],[2,1],inplace=True)
# myData['LIS'].replace(['N','Y'],[2,1],inplace=True)
# myData['MCO_HLVL_PLAN_CD'].replace(['MA','MAPD'],[2,1],inplace=True)
# myData['MCO_PROD_TYPE_CD'].replace(['HMO','LPPO','PFFS','RPPO'],[4,3,2,1],inplace=True)
# myData['Dwelling_Type'].replace(['A','B','C','M','N','P','S','T'],[8,7,6,5,4,3,2,1],inplace=True)
#==============================================================

##replace null with 0
#myData.replace(np.nan,0,inplace=True)

#x = myData.values

#min_max_scaler = preprocessing.MinMaxScaler()
#k=myData.iloc[:,1:447]
#k=k.values
#k_scaled = min_max_scaler.fit_transform(k)
#normalizedDataFrame = pd.DataFrame(k_scaled)
##add ID column to the normalized dataframe
#idcolumn=myData.iloc[:,0]
#ID=idcolumn.to_frame()
#numerical_data=ID.join(normalizedDataFrame)

#numerical_data.to_csv('numerical_data.csv')
#numerical_data = pd.DataFrame(columns=myData.columns, index=myData.index)
#pprintnumerical_data[:100]

#--------------------------------------------------------------------------

##combine the CON_VISIT_XX_QYY variable
baseline='CON_VISIT_'
conXX=list(range(10,34))
XXlist=[]
for ele in conXX:
    XXlist.append(str(ele))
conlist=['01','02','03','04','05','06','07','08','09']   
for ele in XXlist:
    conlist.append(ele)
##create the Quarter line string
Qline=['_Q01','_Q02','_Q03','_Q04']

#calculate sum of four quaters
del conlist[28]
con_list=[]
for i in conlist:
    eles=[]
    for j in Qline:
        eles.append(baseline+i+j)
    myData['CON_'+i]=sum(myData[ele] for ele in eles)
    for ele in eles:
        del myData[ele]

del myData['PCP_ASSIGNMENT']
del myData['Dwelling_Type']
del myData['Num_person_household']
del myData['College']
del myData['Online_purchaser']
del myData['Online_User']
del myData['Pct_above_poverty_line']
del myData['Decile_struggle_Med_lang']
del myData['Home_value']
del myData['Est_Net_worth']
del myData['Est_income']
del myData['Index_Health_ins_influence']
del myData['Population_density_centile_ST']
#pprint(myData[:10])

# myData['SEX_CD'].replace(['F','M'],[0,1],inplace=True)
# myData['ESRD_IND'].replace(['N','Y'],[1,2],inplace=True)
# myData['HOSPICE_IND'].replace(['N','Y'],[1,2],inplace=True)
# myData['PCP_ASSIGNMENT'].replace(['MEMBER SELECTED','ATTRIBUTED','UNATTRIBUTED',None],[3,2,1,0],inplace=True)
# myData['DUAL'].replace(['N','Y'],[2,1],inplace=True)
# myData['INSTITUTIONAL'].replace(['N','Y'],[2,1],inplace=True)
# myData['LIS'].replace(['N','Y'],[2,1],inplace=True)
# myData['MCO_HLVL_PLAN_CD'].replace(['MA','MAPD'],[2,1],inplace=True)
# myData['MCO_PROD_TYPE_CD'].replace(['HMO','LPPO','PFFS','RPPO'],[4,3,2,1],inplace=True)
# myData['Dwelling_Type'].replace(['A','B','C','M','N','P','S','T'],[8,7,6,5,4,3,2,1],inplace=True)
#==============================================================
## deal with CON_VISIT_10_Q02 and POT_VISIT_
for column in list(myData.columns.values):
   if column.find("POT_VISIT_")>-1:
      del myData[column]

myData.replace(np.nan,0,inplace=True)

x = myData.values

min_max_scaler = preprocessing.MinMaxScaler()
k=myData.iloc[:,1:183]
k=k.values
k_scaled = min_max_scaler.fit_transform(k)
normalizedDataFrame = pd.DataFrame(k_scaled)
##add ID column to the normalized dataframe
idcolumn=myData.iloc[:,0]
ID=idcolumn.to_frame()
numerical_data = ID.join(normalizedDataFrame)
# pprint(myData[:10])
#numerical_data_lable = pd.DataFrame(columns = numerical_data.columns, index = myData.index)
header = np.asarray(myData.columns)
headname = list(header)
#my_list = [2, 3, 5, 7, 11]
my_dict = {k for k in headname}
#print(my_dict)
numerical_data.columns = headname
#pprint(numerical_data[:10])
null_value = numerical_data.isnull().sum()
#print(null_value.to_string())
numerical_data.to_csv('sample_numerical_data.csv')

