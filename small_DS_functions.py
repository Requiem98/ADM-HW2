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


def datetime_range(start, end, delta):
    #This function return a list of time intevals
    delta = timedelta(minutes=delta)
    start = datetime(2020, 10, 26, start)
    end = datetime(2020, 10, 26, end)
    dts = list()
    current = start

    while current < end:
        dts.append(current.strftime('%H:%M'))
        dts.append(current.strftime('%H:%M'))
        current += delta
    
    dts = dts[1:(len(dts)-1)]
    t = np.array(dts).reshape((len(dts)//2),2)
    return t 

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
        out = plt.bar(dic.keys(), dic.values());

    finally:
        steam.reset_index(inplace=True)
    
    return out, dic

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
	
def plotBestAuthor_updater(n):
    #Sorting the numbers of updates and take the first n
	authorMOD_sortedHead =  numberOfUpdateForAuthor().sort_values(["Number of update"], ascending=False).head(n)
    
    #Plot the result
	plt.bar(list(map(str,authorMOD_sortedHead.index)), authorMOD_sortedHead["Number of update"]);
	
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
	AuthorTOT["Number of update"] =  numberOfUpdateForAuthor()
	AuthorTOT["Number of not update"] = numberOfNonUpdateReviewsForAuthor()
	
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


	
	
	
steam = pd.read_csv("./steam_reviews.csv", header="infer", nrows=20000, index_col=0, parse_dates=['timestamp_created', 'timestamp_updated', 'author.last_played'], date_parser=dateparse);














