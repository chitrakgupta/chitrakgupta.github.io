
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Read data
dat = pd.read_csv("../data/final_output.csv")
# Read ground information
groundsDat = pd.read_csv("../data/grounds.csv")


# In[2]:

#groundsList = groundsDat['ground']
# Select the matches affected by D/L
DL = dat[dat[' duckworth_lewis']==1]
# Calculate % of matches affected by D/L
percNA = sum(pd.isnull(DL[' ground']))*100./len(dat)
# Select the grounds where D/L came into play
DLgrounds = DL[' ground']


# In[3]:

# Merge ground information with everything else
def mergeGrounds(df, grounds):
    # Get rid of the spaces in the names of grounds
    df.rename(columns={' ground': 'ground'}, inplace=True)
    df['ground'] = df['ground'].str.strip()
    return pd.merge(df, grounds, on='ground')


# In[4]:

DLmerged = mergeGrounds(DL, groundsDat)
allMerged = mergeGrounds(dat, groundsDat)

### Some data was lost here because some games were played at grounds whose ground information is not available
print "Out of ", dat.shape[0], " games, ", allMerged.shape[0], " are being retained"
print "That is ", allMerged.shape[0]*100./dat.shape[0], "% of total matches"


# In[5]:

# Does matches get affected by D/L in all countries that cricket is played in?
totalCountries = len(allMerged['country'].value_counts())
DLcountries = len(DLmerged['country'].value_counts())
plt.bar(range(2),[totalCountries, DLcountries])
plt.xticks([0.5,1.5], ["All", "D/L affected"])
plt.ylabel("Number of countries")
plt.savefig("../figures/02-DL-countries.png")
plt.show()

# Out of 41 countries where cricket is played in, only 24 has had matches affected by D/L so far.


# In[6]:

# Is effect of D/L more on day & night games?

# % of games where D/L came into play which were day & night
DLDN = sum(DLmerged[' day_n_night']==1)*100./len(DLmerged)
# % of all games which were day & night
allDN =sum(allMerged[' day_n_night']==1)*100./len(allMerged)
plt.bar(range(2),[allDN, DLDN])
plt.xticks([0.5,1.5], ["All", "D/L affected"])
plt.ylabel("% of day & night matches")
plt.savefig("../figures/01-DL-DN.png")
plt.show()

# Clearly, day & night games are affected more by D/L


# In[7]:

# Number of D/L affected games by countries
def plotCountries(df, save=0, fName="test.png"):
    countries = df['country'].value_counts()
    names = list(countries.index)
    # Renaming some of the countries to make the labels look better
    if ("United States of America" in names):
        ind = names.index("United States of America")
        names[ind] = "USA"
    if ("United Arab Emirates" in names):
        ind = names.index("United Arab Emirates")
        names[ind] = "UAE"
    if ("Papua New Guinea" in names):
        ind = names.index("Papua New Guinea")
        names[ind] = "PNG"
    if ("Cayman Islands" in names):
        ind = names.index("Cayman Islands")
        names[ind] = "KY"
    # Done
    xVals = np.array(range(len(countries)))
    plt.bar(xVals, countries)
    plt.xticks(xVals+0.5,names,rotation='vertical')
    plt.gcf().subplots_adjust(bottom=0.3)
    if (save):
        plt.savefig("../figures/"+fName) 
    plt.show()


# In[8]:

plotCountries(DLmerged, save=1, fName="03-DL-all-countries.png")

# This seems to suggest most D/L affected matches are played in England


# In[9]:

# Since D/L only applies to limited overs, let us separate these out
# This includes ListA, ODI, T20 and T20I. Excludes Test and First Class matches
limitedDat = dat[(dat[' type_of_match']!='TEST') & (dat[' type_of_match']!='FC')]
limMerged = mergeGrounds(limitedDat, groundsDat)
# Let us see where the matches are played
plotCountries(limMerged,save=1, fName="04-All-countries.png")

# Now we see that most limited over matches are also played in England, so the previous figure was misleading
# Let us see the same thing as the percentage of games played in a given country


# In[10]:

# Percentage of D/L affected games by countries
def plotCountriesPerc(df, dfRef, save=0, fName = "test.png"):
    countries = df['country'].value_counts().sort_index()
    allMatchCountries = dfRef['country'].value_counts().sort_index()
    percDL = (100.0*countries/allMatchCountries).sort_values(ascending=False)
    names = list(percDL.index)
    # Renaming "United States of America" to USA and "United Arab Emirates" to UAE to make the labels look better
    if ("United States of America" in names):
        ind = names.index("United States of America")
        names[ind] = "USA"
    if ("United Arab Emirates" in names):
        ind = names.index("United Arab Emirates")
        names[ind] = "UAE"
    if ("Papua New Guinea" in names):
        ind = names.index("Papua New Guinea")
        names[ind] = "PNG"
    if ("Cayman Islands" in names):
        ind = names.index("Cayman Islands")
        names[ind] = "KY"
    ## Done
    xVals = np.array(range(len(percDL)))
    plt.bar(xVals, percDL)
    plt.xticks(xVals+0.5,names,rotation='vertical')
    plt.ylabel("% of games affected by D/L")
    plt.gcf().subplots_adjust(bottom=0.2)
    if (save):
        plt.savefig("../figures/"+fName) 
    
    plt.show()


# In[11]:

limMergedCountries = limMerged[limMerged['country'].isin(DLmerged['country'])]
plotCountriesPerc(DLmerged,limMergedCountries, save=1, fName='05-DL-countries-percentage.png')
# Now we see a totally different picture.


# In[12]:

# Let us focus on the countries where most games are played.
countries = limMerged['country'].value_counts()
names = list(countries.index)
#print countries[:20]
# Listing the top 20 countries, it seems worthwhile to focus on the countries where number of games > 1000
topCountries = list(countries[countries > 1000].index)
topMergedCountries = limMerged[limMerged['country'].isin(topCountries)]

topDLcountries = DLmerged[DLmerged['country'].isin(topMergedCountries['country'])]
plotCountriesPerc(topDLcountries, topMergedCountries, save=1, fName='06-DL-topCountries-percentage.png')


# In[13]:

# Now let us see which time of the year does matches get affected by D/L?
def plotMonths(df,save=0,fName='test.png'):
    Months = df[' date'].str.split().apply(pd.Series, 1).stack()[:,0]
    monthCounts = Months.value_counts()
    # This is because sometimes the accurate date isn't available
    # So some random words become labeled as "month". But these are listed after the 12 months
    # So selecting the first 12 elements does the trick. Next Section verifies this is justified
    if (len(monthCounts)>12):
        monthCounts = monthCounts[:12]
    
    names = list(monthCounts.index)
    xVals = np.array(range(len(monthCounts)))
    plt.bar(xVals, monthCounts/np.sum(monthCounts))
    plt.xticks(xVals+0.5, names, rotation='vertical')
    if (save):
        plt.savefig("../figures/"+fName)
        
    plt.show()


# In[14]:

if (False):
    Months = allMerged[' date'].str.split().apply(pd.Series, 1).stack()[:,0]
    monthCounts = Months.value_counts()
    
    print "Out of ", np.cumsum(monthCounts)[-1], " matches, month information of ", np.cumsum(monthCounts)[11], "has been accounted for"
    print "That is ", np.cumsum(monthCounts)[11]*100./np.cumsum(monthCounts)[-1], "% of total matches"


# In[15]:

#plotMonths(DLmerged, save=1, fName="DuckworthLewis-months.png")
plotMonths(topDLcountries, save=1, fName="07-DuckworthLewis-top-months.png")
# We don't see a lot of pattern, but maybe time of the year would be related to the country.
# Different countries get rain at different times of the year


# In[16]:

import datetime
def getMonthInfo(df):
    Months = df[' date'].str.split().apply(pd.Series, 1).stack()[:,0]
    df['month'] = Months
    months = {datetime.datetime(2000,i,1).strftime("%b"): i for i in range(1,13)}
    df['month_number'] = df['month'].map(months)


# In[17]:

getMonthInfo(topDLcountries)
getMonthInfo(topMergedCountries)


# In[18]:

import calendar
from matplotlib import cm
def plotCountriesAndMonthsPerc(df, dfRef, save=0, fName='test.png'):
    DLcrossTab = pd.crosstab(df.country, df.month_number)
    allcrossTab = pd.crosstab(dfRef.country, dfRef.month_number)
    DLmatrix = DLcrossTab.as_matrix()
    allmatrix = allcrossTab.as_matrix()
    perc = 100.0*DLmatrix/allmatrix
    realPerc = np.nan_to_num(perc)
    monthNames=[]
    for i in range(1,13):
        monthNames.append(calendar.month_abbr[i])
    
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    mat = ax.matshow(realPerc, cmap=cm.Reds)
    ax.set_xticks(range(12))
    ax.set_xticklabels(monthNames, rotation=90)
    ax.set_yticks(range(realPerc.shape[0]))
    ax.set_yticklabels(list(DLcrossTab.index))
    plt.colorbar(mat)
    plt.gcf().subplots_adjust(left=0.2)
    if (save):
        plt.savefig("../figures/"+fName)
        
    plt.show()


# In[22]:

plotCountriesAndMonthsPerc(topDLcountries, topMergedCountries, save=1, fName="08-CountriesAndMonths.png")


# In[20]:

# What % of matches are won by team that batted first?
def BatFirstResult(df):
    #TossChoiceResult = DLmerged[' win_toss'].to_frame().join(DLmerged[' bat_or_bowl'].to_frame()).join(DLmerged[' win_game'].to_frame())
    TossChoiceResult = df[[' team1', ' team2', ' win_toss', ' bat_or_bowl', ' win_game', ' type_of_match']]
    batFirst = []
    for i in range(TossChoiceResult.shape[0]):
        if (TossChoiceResult[' bat_or_bowl'][i] == 'bat'):
            batFirst.append(TossChoiceResult[' win_toss'][i])
        else:
            if (TossChoiceResult[' win_toss'][i] == TossChoiceResult[' team1'][i]):
                batFirst.append(TossChoiceResult[' team2'][i])
            else:
                batFirst.append(TossChoiceResult[' team1'][i])
            
    TossChoiceResult['bat_first'] = batFirst
    batFirstWin = TossChoiceResult[TossChoiceResult[' win_game'] == TossChoiceResult['bat_first']]
    
    return batFirstWin, sum(TossChoiceResult[' win_game'] == TossChoiceResult['bat_first'])*100./TossChoiceResult.shape[0]


# In[21]:

# Does batting first help in D/L affected matches?
DLresult, DLresultCount = BatFirstResult(DLmerged)
#allDLcountries.index = range(allDLcountries.shape[0])
allResult, allResultCount = BatFirstResult(allMerged)


plt.bar(range(2),[allResultCount, DLresultCount])
plt.xticks([0.5,1.5], ["All", "D/L affected"])
plt.ylabel("% of games won by team batting first")
plt.savefig("../figures/09-DL-battingFirst.png")
plt.show()


# In[ ]:



