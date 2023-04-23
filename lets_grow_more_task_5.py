#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv("globalterrorismdb_0718dist.csv",encoding='latin1')


# In[5]:


df


# In[6]:


df.shape


# In[8]:


df.summary


# In[11]:


df.head()


# # Renaming the column

# In[14]:


df.rename(columns={'iyear':'Year','imonth':'Month','iday':"day",'gname':'Group','country_txt':'Country','region_txt':'Region','provstate':'State','city':'City','latitude':'latitude',
    'longitude':'longitude','summary':'summary','attacktype1_txt':'Attacktype','targtype1_txt':'Targettype','weaptype1_txt':'Weapon','nkill':'kill',
     'nwound':'Wound'},inplace=True)
df1 = df[['Year','Month','day','Country','State','Region','City','latitude','longitude',"Attacktype",'kill',
               'Wound','target1','summary','Group','Targettype','Weapon','motive']]


# In[15]:


df1


# In[17]:


df1.describe()


# In[19]:


df1.isnull().sum()


# In[20]:


data['Wound'] = data['Wound'].fillna(0)
data['kill'] = data['kill'].fillna(0)


# In[21]:


data['Casualities'] = data['kill'] + data['Wound']


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns
year = data['Year'].unique()
years_count = data['Year'].value_counts(dropna = False).sort_index()
plt.figure(figsize = (18,10))
sns.barplot(x = year,
           y = years_count,
           palette = "tab10")
plt.xticks(rotation = 30)
plt.xlabel('Attacking Year',fontsize=10)
plt.ylabel('Attacks Year',fontsize=10)
plt.title('Attacks  Years',fontsize=20)
plt.show()


# In[26]:


pd.crosstab(data.Year, data.Region).plot(kind='area',stacked=True,figsize=(12,10))
plt.title('Terrorist Activities  Region  Year',fontsize=10)
plt.ylabel('Number of Attacks',fontsize=10)
plt.xlabel("Year",fontsize=10)
plt.show()


# In[27]:


attacks = df1.Country.value_counts()[:10]
attacks


# In[29]:


df1.Group.value_counts()[1:5]


# In[32]:


df = data[['Year','kill']].groupby(['Year']).sum()
fig, ax4 = plt.subplots(figsize=(20,10))
df.plot(kind='bar',alpha=0.7,ax=ax4)
plt.xticks(rotation = 50)
plt.title("People Died Due To Attack",fontsize=25)
plt.ylabel("Number of killed peope",fontsize=20)
plt.xlabel('Year',fontsize=20)
top_side = ax4.spines["top"]
top_side.set_visible(False)
right_side = ax4.spines["right"]
right_side.set_visible(False)


# In[38]:


data['City'].value_counts().to_frame().sort_values('City',axis=0,ascending=False).head(10).plot(kind='bar',figsize=(20,10),color='red')
plt.xticks(rotation = 50)
plt.xlabel("City",fontsize=15)
plt.ylabel("Number of attack",fontsize=15)
plt.title("Top 10 most effected city",fontsize=20)
plt.show()


# In[37]:


data[['Attacktype','Wound']].groupby(["Attacktype"],axis=0).sum().plot(kind='bar',figsize=(20,10),color='black')
plt.xticks(rotation=50)
plt.title("Number of wounded  ",fontsize=20)
plt.ylabel('Number of people',fontsize=15)
plt.xlabel('Attack type',fontsize=15)
plt.show()


# In[40]:


data['Group'].value_counts().to_frame().drop('Unknown').head(10).plot(kind='bar',color='pink',figsize=(20,10))
plt.title("Top 10 terrorist group attack",fontsize=20)
plt.xlabel("terrorist group name",fontsize=15)
plt.ylabel("Attack number",fontsize=15)
plt.show()


# In[41]:


kill = data.loc[:,'kill']
print('Number of people killed by terror attack:', int(sum(kill.dropna())))
typeKill = data.pivot_table(columns='Attacktype', values='kill', aggfunc='sum')
typeKill
countryKill = data.pivot_table(columns='Country', values='kill', aggfunc='sum')
countryKill


# In[ ]:




