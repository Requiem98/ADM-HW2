import numpy as np
import pandas as pd
import swifter
import dask
import dask.dataframe as dd
from datetime import *
from collections import *
import matplotlib.pyplot as plt
import warnings

## Generic functions ##

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

## RQ2 functions ##

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

    plt.figure(figsize=(30,5))

        
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
    
## RQ3 functions ##

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

## RQ6 functions ##

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

## RQ7 functions ##

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
