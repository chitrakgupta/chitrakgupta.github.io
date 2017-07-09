
# coding: utf-8

# # Duckworth-Lewis in limited overs cricket matches
# 
# ## Introduction
# Duckworth-Lewis method (https://en.wikipedia.org/wiki/Duckworth%E2%80%93Lewis_method) is used to calculate the modified target score for the team batting second in a cricket match where some overs are lost due to rain interruption. In this blog, I have done taken a close look at such matches, primarily focusing on location (country) and the time of the year when a match is most likely to get affected by D/L.
# 
# The data is available on espncricinfo.com. Scraping the website and converting the data into csv format was done by Gaurav Sood (https://github.com/soodoku/get-cricket-data) and Derek Willis (https://github.com/dwillis/toss-up). I used the two csv files published by these authors on their respective github repositories, where they did an excellent review of the effect of toss in a cricket match (including the effect of toss on D/L affected matches).

# ## Data preparation
# First I read the final output data into a pandas data frame and select the matches which were affected by D/L. Then I read the grounds data into another pandas data frame.

# In[1]:

import pandas as pd
dat = pd.read_csv("../data/final_output.csv")
DL = dat[dat[' duckworth_lewis']==1]
groundsDat = pd.read_csv("../data/grounds.csv")


# Now, I define a function to merge the two final output data and the grounds data. Then do this merging for the "all" data set ("dat") and the subset of matches affected by D/L ("DL"). Some data was lost here because some games were played at grounds whose ground information is not available. However, the percentage of lost data is not very high.

# In[2]:

def mergeGrounds(df, grounds):
    # Get rid of the spaces in the names of grounds
    df2 = df.rename(columns={' ground': 'ground'}, inplace=False)
    df2['ground'] = df2['ground'].str.strip()
    return pd.merge(df2, grounds, on='ground')

DLmerged = mergeGrounds(DL, groundsDat)
allMerged = mergeGrounds(dat, groundsDat)

print "Out of ", dat.shape[0], " games, ", allMerged.shape[0], " are being retained"
print "That is ", "%.2f" % (allMerged.shape[0]*100./dat.shape[0]), "% of total matches"


# ## Results
# The first question I asked is, are day/night games affected by D/L more than day games?

# In[3]:

import matplotlib.pyplot as plt
allDN =sum(allMerged[' day_n_night']==1)*100./len(allMerged)
DLDN = sum(DLmerged[' day_n_night']==1)*100./len(DLmerged)
plt.bar(range(2),[allDN, DLDN])
plt.xticks([0.5,1.5], ["All", "D/L affected"])
plt.ylabel("% of day & night matches")
plt.savefig("../figures/01-DL-DN.png")
plt.show()


# ![title](./figures/01-DL-DN.png)
# Clearly, day & night games are affected more by D/L
# 
# Next, I ask whether there is an advantage to the side batting first in a game affected by D/L. Typically, the target asking rate (runs per over) set for the chasing team in a D/L game is higher than that achieved by the team batting first. This raises the question as to whether the percentage of games won by the team batting first changes significantly in a D/L affected match.

# In[5]:

pd.options.mode.chained_assignment = None
def BatFirstResult(df):
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
            
    TossChoiceResult.loc[:,'bat_first'] = pd.Series(batFirst)
    batFirstWin = TossChoiceResult[TossChoiceResult[' win_game'] == TossChoiceResult['bat_first']]
    
    return batFirstWin, sum(TossChoiceResult[' win_game'] == TossChoiceResult['bat_first'])*100./TossChoiceResult.shape[0]

DLresult, DLresultCount = BatFirstResult(DLmerged)
allResult, allResultCount = BatFirstResult(allMerged)
plt.bar(range(2),[allResultCount, DLresultCount])
plt.xticks([0.5,1.5], ["All", "D/L affected"])
plt.ylabel("% of games won by team batting first")
plt.savefig("../figures/02-DL-battingFirst.png")
plt.show()


# ![title](./figures/02-DL-battingFirst.png)
# 
# We see there is in fact a very little difference (38.66% in all games vs. 45.35% in D/L games).
# 
# Now I focus on how the location affects the likelihood of a match being affected by D/L. Does D/L come into play at all locations (countries)?

# In[6]:

totalCountries = len(allMerged['country'].value_counts())
DLcountries = len(DLmerged['country'].value_counts())
plt.bar(range(2),[totalCountries, DLcountries])
plt.xticks([0.5,1.5], ["All", "D/L affected"])
plt.ylabel("Number of countries")
plt.savefig("../figures/03-DL-countries.png")
plt.show() 


# ![title](./figures/03-DL-countries.png)
# 
# We see that out of 41 countries where cricket is played in, only 24 has had matches affected by D/L so far.
# 
# Next, I looked at how the location (country) affects the number of matches that gets affected by D/L.

# In[7]:

import numpy as np
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

plotCountries(DLmerged, save=1, fName="04-DL-all-countries.png")


# ![title](./figures/04-DL-all-countries.png)
# 
# At first look, it appears that England has the highest instances of matches being affected by D/L. However, this might simply be due to more number of matches being played in England. In other words, highest instances need not mean highest probability.
# 
# So we need to look at the number of matches played per country. Since D/L comes into play only in limited over games, we choose only to focus on the four categories, namely, "ODI", "LISTA", "T20I" and "T20", and leave out "TEST" and "FC".

# In[8]:

limitedDat = dat[(dat[' type_of_match']!='TEST') & (dat[' type_of_match']!='FC')]
limMerged = mergeGrounds(limitedDat, groundsDat)
plotCountries(limMerged,save=1, fName="05-All-countries.png")


# ![title](./figures/05-All-countries.png)
# 
# We see that the hunch was right. The higher occurrence of D/L affected matches in England was simply due to the higher number of matches being played in England. We need to look at the percentage of matches being affected by D/L instead of the number of them.

# In[9]:

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

limMergedCountries = limMerged[limMerged['country'].isin(DLmerged['country'])]
plotCountriesPerc(DLmerged,limMergedCountries, save=1, fName='06-DL-countries-percentage.png')


# ![title](./figures/06-DL-countries-percentage.png)
# 
# Now we see a totally different picture. At this point, let us focus on the countries where most games are played. Let us choose 1000 games as an arbitrary cut-off for the countries we want to include in the analysis.

# In[10]:

countries = limMerged['country'].value_counts()
names = list(countries.index)
topCountries = list(countries[countries > 1000].index)
topMergedCountries = limMerged[limMerged['country'].isin(topCountries)]
topDLcountries = DLmerged[DLmerged['country'].isin(topMergedCountries['country'])]
plotCountriesPerc(topDLcountries, topMergedCountries, save=1, fName='07-DL-topCountries-percentage.png')


# 
# 

# ![title](./figures/07-DL-topCountries-percentage.png)
# 
# We see games in Sri Lanka has the highest probability of being affected by D/L.
# 
# Since D/L is directly linked to the weather, there might be a correlation between the time of the year when the match is being played and the probability of the match being affected by D/L. Let us look at that.

# In[11]:

def plotMonths(df,save=0,fName='test.png'):
    Months = df[' date'].str.split().apply(pd.Series, 1).stack()[:,0]
    monthCounts = Months.value_counts()
    # Sometimes the accurate date isn't available
    # So some random words become labeled as "month". But these are listed after the 12 months
    # So, need to select the first 12 elements does the trick.
    # We shall verifustify in the next section that we aren't losing a whole lot.
    if (len(monthCounts)>12):
        monthCounts = monthCounts[:12]
    
    names = list(monthCounts.index)
    xVals = np.array(range(len(monthCounts)))
    plt.bar(xVals, monthCounts/np.sum(monthCounts))
    plt.xticks(xVals+0.5, names, rotation='vertical')
    if (save):
        plt.savefig("../figures/"+fName)  
    plt.show()
    
plotMonths(topDLcountries, save=1, fName="08-DuckworthLewis-top-months.png")


# ![title](./figures/08-DuckworthLewis-top-months.png)
# 
# We don't see a lot of pattern, but maybe time of the year would be related to the country. Different countries get rain at different times of the year.
# 
# But before proceeding to that, let us verify that we didn't throw out too many matches which had incorrectly formatted date.

# In[12]:

Months = allMerged[' date'].str.split().apply(pd.Series, 1).stack()[:,0]
monthCounts = Months.value_counts()
print "Out of ", np.cumsum(monthCounts)[-1], " matches, month information of ", np.cumsum(monthCounts)[11], "has been accounted for"
print "That is ", "%.2f" % (np.cumsum(monthCounts)[11]*100./np.cumsum(monthCounts)[-1]), "% of total matches"


# In[13]:

import datetime
def getMonthInfo(df):
    Months = df[' date'].str.split().apply(pd.Series, 1).stack()[:,0]
    df.loc[:, 'month'] = pd.Series(Months)
    months = {datetime.datetime(2000,i,1).strftime("%b"): i for i in range(1,13)}
    df.loc[:,'month_number'] = pd.Series(df['month'].map(months))

getMonthInfo(topDLcountries)
getMonthInfo(topMergedCountries)

import calendar
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
def plotCountriesAndMonthsPerc(df, dfRef, save=0, fName='test.png'):
    DLcrossTab = pd.crosstab(df.country, df.month_number)
    allcrossTab = pd.crosstab(dfRef.country, dfRef.month_number)
    DLmatrix = DLcrossTab.as_matrix()
    allmatrix = allcrossTab.as_matrix()
    perc = 100.0 * np.true_divide(DLmatrix, allmatrix, where=(allmatrix!=0))
    perc[allmatrix == 0] = 0
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mat, cax = cax)
    plt.gcf().subplots_adjust(left=0.2)
    if (save):
        plt.savefig("../figures/"+fName)
    plt.show()
    
plotCountriesAndMonthsPerc(topDLcountries, topMergedCountries, save=1, fName="09-CountriesAndMonths.png")


# ![title](./figures/09-CountriesAndMonths.png)
# 
# We see that the highest probability of D/L coming into play is in matches played in India in August and in New Zealand in September, followed by those played in New Zealand (November), Bangladesh (Jun and September), Sri Lanka (December) and West Indies (July).
