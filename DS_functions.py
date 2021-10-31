import numpy as np
import pandas as pd
import swifter
import dask
import dask.dataframe as dd
from datetime import *
from collections import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import scipy
from scipy.stats import t
from sklearn import linear_model as  lm
import seaborn as sns

"""============================================================================================================================"""

"""Generic functions""" 

"""============================================================================================================================"""

def dateparse(time_as_a_unix_timestamp):
    return pd.to_datetime(time_as_a_unix_timestamp, unit='s')

# This function return a list of time intevals on 24h
# Input: 
#     startM: starting minutes
#     endM: ending minutes
#
def datetime_range24(startM=0, endM=0, delta=60): 
    delta = timedelta(minutes=delta)
    start = datetime(2020, 10, 26, 0, startM)
    end = datetime(2020, 10, 27, 0, endM)
    dts = list()
    current = start

    while current <= end:
        dts.append(current.strftime('%H:%M'))
        dts.append(current.strftime('%H:%M'))
        current += delta
    
    dts = dts[1:(len(dts)-1)]
    t = np.array(dts).reshape((len(dts)//2),2)
    return t

#function that map the values of a plot into colors
def plot_color(values, size=(30,5), coef=0.2):
    plt.rcParams['axes.facecolor'] = 'ivory'
    plt.figure(figsize=size)
    plt.grid()
    my_cmap = plt.get_cmap('Greys')
    my_norm = plt.Normalize(vmin=0, vmax=max(values)*coef)
    colormap=my_cmap(my_norm(values))
    
    return colormap

"""============================================================================================================================"""

"""RQ1 functions""" 

"""============================================================================================================================"""

#Compute the heatmap of null values for each of the columns of the dataset
"""da modificare"""
def nullHeatMap():
    return sns.heatmap(steam[[4,18,19,20,21]].isnull(),cbar=True,yticklabels=False,cmap = 'viridis')

#Functions that shows the main characteristics of quantitative variables
# Output: 
#    -Dataframe
def statisticalIndexes():
    pd.set_option('float_format', '{:.2f}'.format)
    data = steam.describe().iloc[:,[2, 3, 4, 5, 7, 8, 9, 10, 11]]
    return data

#Functions that shows the main characteristics of quantitative variables (plot)
# Output: 
#    -boxplot
def plot_statisticalIndexes():
    box = steam.iloc[:,[8,9,10,11,16,17]].plot.box(subplots=True,figsize=(20,8));
    return box

def histOfTimeStampCreated(delta):

    #Get dates from the dataset and set it all at the same day
    dates = steam["timestamp_created"]

    #convert all the dates in to float
    mpl_data = mdates.date2num(dates)
       
    #create the bins on range  
    binns = np.arange(min(mpl_data), max(mpl_data)+delta , delta) 
    
    #plot the result
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches((30,5))
    n, bins, temp = ax.hist(mpl_data, bins=binns , color="k")
    locator = mdates.AutoDateLocator()
    ax.grid()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    plt.xlabel("Time", labelpad=25.0, size="xx-large")
    plt.ylabel("Number of reviews", labelpad=25.0, size="xx-large")
    plt.title("Number of reviews during time")
    
    #create a colormap for the value
    my_cmap = plt.get_cmap('Greys')
    col = (n-n.min())/(n.max()-n.min())
 
    for c, p in zip(col, temp):
        plt.setp(p, 'facecolor', my_cmap(c))
        
    
    #create the legend with the max valueb
    temp = mdates.num2date(bins)
    maxvalSX = temp[np.where(n==max(n))[0][0]]
    maxvalDX = mdates.num2date(bins[np.where(n==max(n))[0][0]]+59)
    maxval = str(maxvalSX) + "|" + str(maxvalDX)
    plt.legend(["max value = " + maxval])


    return n, bins, maxval

def reviewsPieCharts():

    sp=steam[['steam_purchase','review_id']].groupby('steam_purchase').count()
    rec=steam[['recommended','review_id']].groupby('recommended').count()
    rff=steam[['received_for_free','review_id']].groupby('received_for_free').count()
    wea=steam[['written_during_early_access','review_id']].groupby('written_during_early_access').count()

    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(15,15)
    ax[0, 0].pie(sp.review_id, autopct='%1.1f%%', colors = ["dimgray", "k"], shadow=False, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'}, textprops={'size': 'x-large', "color": "w"}, labels=["Not purchased","purchased"], labeldistance=None);
    ax[0, 1].pie(rec.review_id, autopct='%1.1f%%', colors = ["dimgray", "k"], shadow=False, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'}, textprops={'size': 'x-large', "color": "w"}, labels=["Not recommended","recommended"], labeldistance=None);
    ax[1, 0].pie(rff.review_id, autopct='%1.1f%%', colors = ["dimgray", "k"], shadow=False, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'}, textprops={'size': 'x-large', "color": "w"}, labels=["Not free","free"], labeldistance=None);
    ax[1, 1].pie(wea.review_id, autopct='%1.1f%%', colors = ["dimgray", "k"], shadow=False, wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'}, textprops={'size': 'x-large', "color": "w"}, labels=["Not written durin early access","written during early access"], labeldistance=None);

    for x in ax.flat:
        x.legend();
        
    return sp, rec, rff, wea


"""============================================================================================================================"""

"""RQ2 functions""" 

"""============================================================================================================================"""


def numberOfReviewsByApplication(n=0):
    #Create a slice of the dataframe with the column "app_name" and "review_id" and than group it by "app_name"
    #Sorting the result and take n rows
    
    #If i want just a part of the dataset...
    warnings.filterwarnings("ignore")
    if(n != 0):
        numbersOfReviewsByApp = steam[["app_name", "review_id"]].groupby(["app_name"]).count().sort_values(["review_id"], ascending=True).tail(n)
    #else...
    else:
        numbersOfReviewsByApp = steam[["app_name", "review_id"]].groupby(["app_name"]).count().sort_values(["review_id"], ascending=True)

    #get the data for the barplot
    height = numbersOfReviewsByApp["review_id"].array
    val = numbersOfReviewsByApp.index

    #plot the result
    #fancy color =) 
    if(n != 0 and n < 20):
        
        cmap = plot_color(values=height, coef=1)
        
    else:
        cmap = plot_color(values=height, size=(20,60))

    
    plt.barh(val, height, color=cmap);
    plt.xticks(rotation="vertical")
    plt.xlabel("Number of reviews", labelpad=25.0, size="xx-large")
    plt.ylabel("Videogames", labelpad=25.0, size="xx-large")
    plt.title("Number of reviews by Game")
        
    
    return numbersOfReviewsByApp


def scoreOfApps(n=0):
    #Create a slice of the dataframe with the column "app_name" and "weighted_vote_score" and than group it by "app_name"
    #Get just the max of each group and than sort
    
    #If i want just a part of the dataset...
    if(n != 0):
        appScore = steam[["app_name", "weighted_vote_score"]].groupby(["app_name"]).max().sort_values(["weighted_vote_score"], ascending=False).head(n)
        
    #else...
    else:
        appScore = steam[["app_name", "weighted_vote_score"]].groupby(["app_name"]).max().sort_values(["weighted_vote_score"], ascending=False)
    
    return appScore

def reccomendedApp_purchase_free(h=0, t=0):
    
    #If i want all the dataset t = 0
    if(t == 0):
        reccomendedApp_Final = steam[["app_name", "recommended", "received_for_free", "steam_purchase"]].groupby(["app_name"]).sum()
        temp = steam[["app_name", "review_id"]].groupby("app_name").count()
        reccomendedApp_Final["TOT reviews"] = temp
        reccomendedApp_Final["percentage of reccommended"] = reccomendedApp_Final.swifter.apply(lambda row: row["recommended"]/row["TOT reviews"], axis=1)
        reccomendedApp_Final = reccomendedApp_Final.sort_values(["percentage of reccommended", "TOT reviews"], ascending=False)
    #else i want the first h and the last t
    else:
        
        reccomendedApp = steam[["app_name", "recommended", "received_for_free", "steam_purchase"]].groupby(["app_name"]).sum()
        temp = steam[["app_name", "review_id"]].groupby("app_name").count()
        reccomendedApp["TOT reviews"] = temp
        
        reccomendedApp["percentage of reccommended"] = reccomendedApp.swifter.apply(lambda row: row["recommended"]/row["TOT reviews"], axis=1)
        
        reccomendedApp = reccomendedApp.sort_values(["percentage of reccommended", "TOT reviews"], ascending=False)
        
        reccomendedApp_Final = reccomendedApp.head(h)
        reccomendedApp_Final = pd.concat([reccomendedApp_Final, reccomendedApp.tail(t)])
        
    return reccomendedApp_Final
    
"""============================================================================================================================"""

"""RQ3 functions""" 

"""============================================================================================================================"""

def histOfDayTime(delta):

    #float value rappresenting a minute on the matplotlib conversion
    m = 0.0006944444444444444
    #Get dates from the dataset and set it all at the same day
    dates = steam["timestamp_created"].swifter.progress_bar(False).apply(lambda row: row.replace(year=1970, month=1, day=1, second=0))

    #convert all the dates in to float
    mpl_data = mdates.date2num(dates)
       
    #create the bins on range 0.0 (1970:1:1 0:00) to 1+m (1970:1:2 0:01) 
    binns = np.arange(0.0, 1.0+m , delta*m) 
    
    #plot the result
    plt.rcParams['axes.facecolor'] = 'ivory'
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches((20,5))
    n, bins, temp = ax.hist(mpl_data, bins=binns , color="k")
    locator = mdates.AutoDateLocator()
    ax.grid()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xlabel("Time intervals", labelpad=25.0, size="xx-large")
    plt.ylabel("Number of reviews", labelpad=25.0, size="xx-large")
    
    #create a colormap for the value
    my_cmap = plt.get_cmap('Greys')
    col = (n-n.min())/(n.max()-n.min())
 
    for c, p in zip(col, temp):
        plt.setp(p, 'facecolor', my_cmap(c))
        
    
    #create the legend with the max valueb
    temp = mdates.num2date(bins)
    maxval = temp[np.where(n==max(n))[0][0]].time()
    plt.legend(["max value = " + str(maxval)])


    return n, bins, maxval



def numbersOfReviewByTime(interv):
    #Setting the index timestamp_created in order to use between_time function
    steam.set_index("timestamp_created", inplace=True)
    
    #The index must be reseted in any case so.... try and finnaly!
    try:
        
        #Create an orderedDict
        dic = OrderedDict()
        
        #Adding to the keys of orderedDict the intervals and in the values the number of rows of between_time
        for x in interv:    
            dic[x[0] + "-" + x[1]] = len(steam.between_time(x[0], x[1]))
    
        #plot the result
        cmap = plot_color(list(dic.values()), coef=1)
        
        plt.bar(dic.keys(), dic.values(), color=cmap);

        plt.xticks(rotation="vertical")
        plt.xlabel("Time intervals", labelpad=25.0, size="xx-large")
        plt.ylabel("Number of reviews", labelpad=25.0, size="xx-large")

    finally:
        steam.reset_index(inplace=True)
    
    return dic

"""============================================================================================================================"""

"""RQ4 functions""" 

"""============================================================================================================================"""

# function that receives as parameters both the name of a data set and a list of languages’ 
# names and returns a data frame filtered only with the reviews written in the provided languages.

def filterByLang(data, languages):
    
    data.set_index("language", inplace=True)
    
    try:
        out = data.loc[languages[0]]
     
        for lang in languages[1:]:
            temp = data.loc[lang]
            out = pd.concat([out, temp])
    
    finally:
        data.reset_index(inplace=True)
    
    return out

#function that return a dataframe containing the numbers of reviews grouped by languages sorted
def reviewsByLanguage(n):
    return steam[["language", "review_id"]].groupby(["language"], as_index=False).count().sort_values("review_id", ascending=False).head(n)

# function that plot the result of "reviewsByLanguage" function
def plot_reviewsByLanguage():
    revByLang =  reviewsByLanguage(28)
    
    #get the data for the barplot
    height = revByLang["review_id"].array
    val = revByLang["language"].array

    #plot the result
    #fancy color =)    
    cmap = plot_color(values=height, size=(20,10), coef=0.125)
    plt.bar(val, height, color=cmap);
    
    plt.xticks(rotation="vertical")
    plt.xlabel("Number of reviews", labelpad=25.0, size="xx-large")
    plt.ylabel("Languages", labelpad=25.0, size="xx-large")
    plt.title("Number of reviews by Language")
    
    return revByLang

#function that return a dataframe with the following columns:
#   ||LANGUAGE||VOTES_FUNNY||TOT REVIEWS||PERCENTAGE||
# 
#  -Language: the language of the reviews (ordered by the most used)
#  -Votes_funny: the count of the reviews that received at least one vote as funny
#  -Tot reviews: the total count of reviews
#  -percentage: the percentage of reviews that received at least one vote as funny

def percentageOfFunny(n):
    
    #get the top n language and filter the dataset by those 
    filteredByLang = filterByLang(steam, reviewsByLanguage(n)["language"].array)
    
    #get the revies with at least one vote as funny
    warnings.filterwarnings("ignore")
    ones = filteredByLang[["votes_funny"]]
    ones = ones[ones.votes_funny != 0]
    #Group it by language
    ones = ones.groupby(["language"]).count()
    #get the total number of reviews by language
    tot = filteredByLang[["votes_funny"]].groupby(["language"]).count()
    #Initialize the final dataset with the ones
    final = ones
    #Adding the numbers of reviews as column
    final["tot reviews"] = tot
    #Compute the percentage and add it to the dataset
    final["percentage"] = final.swifter.apply(lambda row: row["votes_funny"]/row["tot reviews"], axis=1)
    
    return final

#function that return a dataframe with the following columns:
#   ||LANGUAGE||VOTES_HELPFUL||TOT REVIEWS||PERCENTAGE||
# 
#  -Language: the language of the reviews (ordered by the most used)
#  -Votes_helpful: the count of the reviews that received at least one vote as helpful
#  -Tot reviews: the total count of reviews
#  -percentage: the percentage of reviews that received at least one vote as helpful

def percentageOfHelpful(n):
    
    #get the top n language and filter the dataset by those 
    filteredByLang = filterByLang(steam, reviewsByLanguage(n)["language"].array)
    
    #get the revies with at least one vote as helpful
    warnings.filterwarnings("ignore")
    ones = filteredByLang[["votes_helpful"]]
    ones = ones[ones.votes_helpful != 0]
    #Group it by language
    ones = ones.groupby(["language"]).count()
    #get the total number of reviews by language
    tot = filteredByLang[["votes_helpful"]].groupby(["language"]).count()
    #Initialize the final dataset with the ones
    final = ones
    #Adding the numbers of reviews as column
    final["tot reviews"] = tot
    #Compute the percentage and add it to the dataset
    final["percentage"] = final.swifter.apply(lambda row: row["votes_helpful"]/row["tot reviews"], axis=1)
    
    return final

"""============================================================================================================================"""

"""RQ5 functions""" 

"""============================================================================================================================"""


#Return a dataframe with the steam_id as index and the number of reviews of the author as column
def mostPopularReviewers_local(n):
    return steam[["author.steamid", "review_id"]].groupby(["author.steamid"]).count().sort_values("review_id",ascending=False).head(n)

"""def mostPopularReviewers_global(n):
    return steam[["author.steamid", "author.num_reviews"]].groupby(["author.steamid"]).first().sort_values("author.num_reviews",ascending=False).head(n)"""

#Plot the result fo the "mostPopularReviews" function
def mostPopularReviewers_plot(n):
    
    #get the data from the "mostPopularReviews" function
    data = mostPopularReviewers_local(n)
    
    #get the values and the height of the values from the dataset
    height = data["review_id"].array
    val = list(map(str,data.index))

    #map the colors with the height
    cmap = plot_color(height, coef=1)

    
    plot = plt.bar(val, height, color=cmap)
    
    plt.xlabel('author.steamid', labelpad=25.0, size="xx-large")
    plt.ylabel('Number of reviews', labelpad=25.0, size="xx-large")
    plt.title('Reviews by author')

    if(n > 10):
        plt.xticks(rotation="vertical")
        
    return plot

#function that return a list of games that the most popular reviewer reviewed
def reviewedApplicationFromTheMostPopularReviewer():
    return steam[steam['author.steamid']==mostPopularReviewers(1).index[0]].app_name

def percentageOfReceived():
    
    tot_Reviews = steam[["author.steamid", "review_id"]].groupby(["author.steamid"], as_index=False).count()
    
    tot_Reviews = tot_Reviews[tot_Reviews["author.steamid"] == mostPopularReviewers_local(1).index[0]]
    
    freeAndPurchase = steam[['author.steamid', "steam_purchase", "received_for_free"]].groupby('author.steamid', as_index=False).sum()
    
    freeAndPurchase = freeAndPurchase[freeAndPurchase['author.steamid']==mostPopularReviewers_local(1).index[0]]
    
    freeAndPurchase["tot reviews"] = tot_Reviews["review_id"]
    
    percPurchase = freeAndPurchase[["steam_purchase", "received_for_free", "tot reviews"]].swifter.apply(lambda row: row["steam_purchase"]/(row["tot reviews"]),axis=1)
    
    percFree = freeAndPurchase[["steam_purchase", "received_for_free", "tot reviews"]].swifter.apply(lambda row: row["received_for_free"]/(row["tot reviews"]),axis=1)
    
    final = freeAndPurchase
    
    final["percentage_of_purchase"] = percPurchase
    final["percentage_of_free"] = percFree
    
    return final

#function that return a dataframe with the following columns:
#   ||RECOMMENDED||STEAM_PURCHASE||RECEIVED_FOR_FREE||
# 
#  -recommended: If the apps are recommended
#  -steam_purchase: number of purchased app that are reccomanded or not
#  -received_for_free: number of app received for free that are reccomanded or not 
def recommendedApp_purchasedAndFree():
    data = steam[['author.steamid', "recommended", "steam_purchase", "received_for_free"]]
    data = data[data['author.steamid']==mostPopularReviewers_local(1).index[0]]
    return data.groupby("recommended", as_index=False).sum(numeric_only=True).drop(["author.steamid"], axis=1)


"""============================================================================================================================"""

"""RQ6 functions""" 

"""============================================================================================================================"""

def timedelta_updated_created():
    return steam[["timestamp_created", "timestamp_updated"]].swifter.apply(lambda row:row["timestamp_updated"] - row["timestamp_created"], axis=1)

def averageTimeBeforeUpdate():
    t = timedelta_updated_created()
    return str(t.describe()["mean"])[0:13]

def numberOfUpdateByAuthor():
	#Get the author.steamID column
	author_steamId_AND_timedelta = steam[["author.steamid"]]
	
    #Add to the taken column the timedelta column
	warnings.filterwarnings("ignore")
	author_steamId_AND_timedelta["Number of update"] = timedelta_updated_created()
	
    #Get just the timedelta different form 0
	authorMOD = author_steamId_AND_timedelta.loc[author_steamId_AND_timedelta["Number of update"] != "0", ["author.steamid", "Number of update"]]
    #Group timedelta by author and count
	authorMOD = authorMOD.groupby(["author.steamid"]).count()
	
	return authorMOD
	
def plotBestAuthor_updater(n = 1):
    #Sorting the numbers of updates and take the first n
    authorMOD_sortedHead = numberOfUpdateByAuthor().sort_values(["Number of update"], ascending=False).head(n)
    
    #map the values with colors
    cmap = plot_color(authorMOD_sortedHead["Number of update"].array, coef=1)
  
    plt.bar(list(map(str,authorMOD_sortedHead.index)), authorMOD_sortedHead["Number of update"], color=cmap);

    plt.xlabel("Steam ID", labelpad=25.0, size="xx-large")
    plt.ylabel("Number of updates", labelpad=25.0, size="xx-large")
    if(n > 8):
        plt.xticks(rotation="vertical")
	
    return authorMOD_sortedHead

def numberOfNonUpdateReviewsByAuthor():
	#Get the author.steamID column
	author_steamId_AND_timedelta = steam[["author.steamid"]]
	
    #Add to the taken column the timedelta column
	warnings.filterwarnings("ignore")
	author_steamId_AND_timedelta["Number of not update"] = timedelta_updated_created()
	
	#Get just the timedelta equals to 0
	authorNOMOD = author_steamId_AND_timedelta.loc[author_steamId_AND_timedelta["Number of not update"] == "0", ["author.steamid", "Number of not update"]]
    #Group timedelta by author and count
	authorNOMOD = authorNOMOD.groupby(["author.steamid"]).count()
	
	return authorNOMOD

def numberOfUpdateAndNonUpdateByAuthor(n):
	#Get the author.steamID column
	author_steamId_AND_timedelta = steam[["author.steamid"]]
	
    #Add to the taken column the timedelta column
	warnings.filterwarnings("ignore")
	author_steamId_AND_timedelta["TOT"] = timedelta_updated_created()
	
    #Group timedelta by author and count
	AuthorTOT = author_steamId_AND_timedelta.groupby(["author.steamid"]).count()
    
    #Adding the Numbers of updates and numbers of not updates
	AuthorTOT["Number of update"] =  numberOfUpdateByAuthor()
	AuthorTOT["Number of not update"] = numberOfNonUpdateReviewsByAuthor()
	
    #compute the percentage of updating of the authors
	perc = AuthorTOT.apply(lambda row: (row["Number of update"]/row["TOT"])*100,axis=1)
	
    #adding the percentage to the dataframe
	AuthorTOT["Update percentage"] = perc
	
    #Sort and return the first n
	return AuthorTOT.sort_values(["Update percentage", "TOT"], ascending=False).head(n)  

"""============================================================================================================================"""

"""RQ7 functions""" 

"""============================================================================================================================"""

# A
def probabilityQuestion1():
	#NUmber of total cases
	tot = steam[["weighted_vote_score"]].shape[0]
	
	#Number of favorable cases
	fav = steam[(steam["weighted_vote_score"] == 0.5)].shape[0]
	
	#Computing the probability...
	p = fav/tot
	
	return p

#Altro
def probabilityQuestion2():
	#NUmber of total cases
	tot = steam[(steam["weighted_vote_score"] > 0.5)].shape[0]
	
	#Number of favorable cases
	fav = steam[(steam.votes_funny > 0) & (steam.weighted_vote_score > 0.5)].shape[0]
	
	#Computing the probability...
	p = fav/tot

	return p

# C
def probabilityQuestion3():
	#NUmber of total cases
	tot = steam[["weighted_vote_score"]].shape[0]
	
	#Number of favorable cases
	fav = steam[(steam["weighted_vote_score"] > 0.5)].shape[0]
	
	#Computing the probability...
	p = fav/tot

	return p

# F
def probabilityQuestion4():
	#NUmber of total cases
	tot = steam[["weighted_vote_score"]].shape[0]
	
	#Number of favorable cases
	fav = steam[(steam["votes_funny"] > 0)].shape[0]
	
	#Computing the probability...
	p = fav/tot

	return p


# F | (C u A)
def probabilityQuestion5():
    #NUmber of total cases
    tot = steam[(steam["weighted_vote_score"] >= 0.5)].shape[0]
    #Number of favorable cases
    fav = steam[(steam.votes_funny > 0) & (steam["weighted_vote_score"] >= 0.5)].shape[0]
    
    #Computing the probability...
    p = fav/tot
    
    return p



#Load the dataset and parse the dates
steam = pd.read_csv("./steam_reviews.csv", header="infer", parse_dates=['timestamp_created', 'timestamp_updated', 'author.last_played'], date_parser=dateparse, index_col=0)

steam['author.playtime_last_two_weeks']=pd.to_timedelta(steam['author.playtime_last_two_weeks'], unit='m')
steam['author.playtime_forever']=pd.to_timedelta(steam['author.playtime_forever'], unit='m')
steam['author.playtime_at_review']=pd.to_timedelta(steam['author.playtime_at_review'], unit='m')

#clean the dataset from wrong data
steam = steam[steam["review_id"] != 51047390]
steam = steam[steam["review_id"] != 61995164]
steam = steam[steam["review_id"] != 62108069]