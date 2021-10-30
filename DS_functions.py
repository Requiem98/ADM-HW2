import numpy as np
import pandas as pd
import swifter
import dask
import dask.dataframe as dd
from datetime import *
from collections import *
import matplotlib.pyplot as plt
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
#    -boxplot
def statisticalIndex():
    data = round(steam.iloc[:,[8,9,10,11,16,17,18,19,20]].describe(),3)
    box = steam.iloc[:,[8,9,10,11,16,17]].plot.box(subplots=True,figsize=(20,8));
    return data, box


"""============================================================================================================================"""

"""RQ2 functions""" 

"""============================================================================================================================"""


def numbersOfReviewsByApplication(n=0):
    #Create a slice of the dataframe with the column "app_name" and "review_id" and than group it by "app_name"
    #Sorting the result and take n rows
    
    #If i want just a part of the dataset...
    if(n != 0):
        numbersOfReviewsByApp = steam[["app_name", "review_id"]].groupby(["app_name"]).count().sort_values(["review_id"], ascending=True).tail(n)
    #else...
    else:
        numbersOfReviewsByApp = steam[["app_name", "review_id"]].groupby(["app_name"]).count().sort_values(["review_id"], ascending=True)

    #get the data for the barplot
    height = numbersOfReviewsByApp["review_id"].array
    val = numbersOfReviewsByApp.index

    #fancy color =) 
    my_cmap = plt.get_cmap('Greys')
    my_norm = plt.Normalize(vmin=0,vmax=(numbersOfReviewsByApp["review_id"].max())*0.2)

    #plot the result

    plt.figure(figsize=(30,30))
    plt.grid()
    plt.xticks(rotation="vertical")
    plt.xlabel("Number of reviews", labelpad=25.0, size="xx-large")
    plt.ylabel("Videogames", labelpad=25.0, size="xx-large")
        
    plt.barh(val, height, color=my_cmap(my_norm(height)));
    
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

def raccomendedApp_purchase_free(h=0, t=0):
    
    #If i want all the dataset t = 0
    if(t == 0):
        raccomendedApp = steam[["app_name", "recommended", "received_for_free", "steam_purchase"]].groupby(["app_name"]).sum().sort_values(["recommended"], ascending=False)
    
    #else i want the first h and the last t
    else:
        
        #get the first h
        raccomendedApp = raccomendedApp = steam[["app_name", "recommended", "received_for_free", "steam_purchase"]].groupby(["app_name"]).sum().sort_values(["recommended"], ascending=False).head(5)
    
        #get the last t and concat it to the dataframe
        raccomendedApp = pd.concat([raccomendedApp, steam[["app_name", "recommended", "received_for_free", "steam_purchase"]].groupby(["app_name"]).sum().sort_values(["recommended"], ascending=False).tail(t)])
    
    return raccomendedApp
    
"""============================================================================================================================"""

"""RQ3 functions""" 

"""============================================================================================================================"""

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
        my_cmap = plt.get_cmap('Greys')
        my_norm = plt.Normalize(vmin=max(dic.values())*0.2)
        plt.figure(figsize=(30,5))
        plt.bar(dic.keys(), dic.values(), color=my_cmap(my_norm(list(dic.values()))));
        plt.grid()
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
    ones.where(filteredByLang.votes_funny != 0, inplace=True)
    ones = ones.dropna()
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
    ones.where(filteredByLang.votes_funny != 0, inplace=True)
    ones = ones.dropna()
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
def mostPopularReviewers(n):
    return steam[["author.steamid", "author.num_reviews"]].groupby(["author.steamid"]).first().sort_values("author.num_reviews",ascending=False).head(n)

#Plot the result fo the "mostPopularReviews" function
def mostPopularReviewers_plot(n):
    
    #get the data from the "mostPopularReviews" function
    data = mostPopularReviewers(n)
    
    #get the values and the height of the values from the dataset
    height = data["author.num_reviews"].array
    val = list(map(str,data.index))

    #map the colors with the height
    my_cmap = plt.get_cmap('Greys')
    my_norm = plt.Normalize(vmin=0,vmax=(data["author.num_reviews"].max())*0.2)
    
    plt.figure(figsize=(30,5))
    
    plot = plt.bar(val, height, color=my_cmap(my_norm(height)));
    plt.xlabel('author.steamid', labelpad=25.0, size="xx-large")
    plt.ylabel('Number of reviews', labelpad=25.0, size="xx-large")
    plt.title('Reviews by author')
    plt.grid()
    if(n > 10):
        plt.xticks(rotation="vertical")
        
    return plot

#function that return a list of games that the most popular reviewer reviewed
def reviewedApplicationFromTheMostPopularReviewer():
    return steam[steam['author.steamid']==mostPopularReviewers(1).index[0]].app_name

def percentageOfReceived():
    
    freeAndPurchase = steam[['author.steamid', "steam_purchase", "received_for_free"]].groupby('author.steamid', as_index=False).sum()
    
    freeAndPurchase = freeAndPurchase[freeAndPurchase['author.steamid']==mostPopularReviewers(1).index[0]]
    
    percPurchase = freeAndPurchase[["steam_purchase", "received_for_free"]].apply(lambda row: row["steam_purchase"]/(row["steam_purchase"] + row["received_for_free"]),axis=1)
    
    percFree = freeAndPurchase[["steam_purchase", "received_for_free"]].apply(lambda row: row["received_for_free"]/(row["steam_purchase"] + row["received_for_free"]),axis=1)
    
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
    data = data[data['author.steamid']==mostPopularReviewers(1).index[0]]
    return data.groupby("recommended", as_index=False).sum(numeric_only=True).drop(["author.steamid"], axis=1)


"""============================================================================================================================"""

"""RQ6 functions""" 

"""============================================================================================================================"""

def timedelta_updated_created():
    return steam[["timestamp_created", "timestamp_updated"]].swifter.apply(lambda row:row["timestamp_updated"] - row["timestamp_created"], axis=1)

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

    my_cmap = plt.get_cmap('Greys')
    my_norm = plt.Normalize(vmin=(authorMOD_sortedHead["Number of update"].max())*0.2)
    
    plt.figure(figsize=(30,5))
    plt.bar(list(map(str,authorMOD_sortedHead.index)), authorMOD_sortedHead["Number of update"], color=my_cmap(my_norm(authorMOD_sortedHead["Number of update"].array)));
    plt.grid()
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

def probabilityQuestion1():
	#NUmber of total cases
	tot = steam[["weighted_vote_score"]].shape[0]
	
	#Number of favorable cases
	fav = steam[(steam["weighted_vote_score"] == 0.5)].shape[0]
	
	#Computing the probability...
	p = fav/tot
	
	return p

def probabilityQuestion2():
	#NUmber of total cases
	tot = steam[(steam["weighted_vote_score"] > 0.5)].shape[0]
	
	#Number of favorable cases
	fav = steam[(steam["votes_funny"] > 0)].shape[0]
	
	#Computing the probability...
	p = fav/tot

	return p

def probabilityQuestion3():
	#NUmber of total cases
	tot = steam[["weighted_vote_score"]].shape[0]
	
	#Number of favorable cases
	fav = steam[(steam["votes_funny"] > 0)].shape[0]
	
	#Computing the probability...
	p = fav/tot

	return p



steam = pd.read_csv("./steam_reviews.csv", header="infer", parse_dates=['timestamp_created', 'timestamp_updated', 'author.last_played'], date_parser=dateparse, index_col=0)

steam['author.playtime_last_two_weeks']=pd.to_timedelta(steam['author.playtime_last_two_weeks'], unit='m')
steam['author.playtime_forever']=pd.to_timedelta(steam['author.playtime_forever'], unit='m')
steam['author.playtime_at_review']=pd.to_timedelta(steam['author.playtime_at_review'], unit='m')