'''
@author Mohammad Al Jamous_s0572984
'''

''' Impoting required Libraries'''
# -----------------------------
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from yellowbrick.cluster import KElbowVisualizer
from tabulate import tabulate
from itertools import product
from sklearn.neighbors import NearestNeighbors #####
from timeit import default_timer as timer
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings

#%%
def get_center(clr):
  """
  calculate the centroid of a cluster 
  Args:
    cluster coordinates, nx2 array-like (array, list of lists, etc) 
    n is the number of points(latitude, longitude)in the cluster.
  Return:
    centroid of the cluster
    
  """
  cluster_array = np.asarray(clr)
  center = cluster_array.mean(axis = 0)
  return center


#%%
def check_prepare_ordata(csv_data):
    '''
    Parameters
    ----------
    csv_data : file 
        this file contains csv data.

    Returns
    -------
    None.

    '''
    
    # Read data from csv-file 
    print(" --------------------- ")
    print(" 1. data reading ...  ")
    print(" --------------------- ")
    
    data = pd.read_csv(csv_data, encoding= 'unicode_escape') 
    
    print('')
    print(" --------------------- ")
    print(" 2. data analysis ...  ")
    print(" --------------------- ")
    
    
    # # Data infos
    print('Data shape: There are {} rows and {} columns '.format(data.shape[0],data.shape[1]))
    print("")
    print('Columns names:', list(data.columns.values.tolist()))
    print("")
    print("First five samples: ")
    print(data.head())
    print("")
    print("Summary statistics of variables: ") 
    print(data.describe())
    print("")
    print("Check missing data: ") # check for missing data 
    if (data.isnull().sum().sum() == 0):
        print("No missing data as shown here:") 
        print(data.isnull().sum()) 
    else:
        print("There are missing data.") 
        print(data.isnull().sum())
        print("Missing data are in the following columns:") 
        print(data.columns[data.isna().any()])
    
    print(np.sum(data.isnull()), "\n")
    print("Percentage of customers missing: ", round(data['CustomerID'].isnull().sum() * 100 / len(data),2),"%" )
    
    # Checking for duplicates if any present
    #print("Shape before dropping duplicates", data.shape)
    print("Number of duplicated rows: ",data.duplicated().value_counts())
    # drop duplicates
    data = data.drop_duplicates()
    #print("Shape after dropping duplicates", data.shape)
       
    # drop rows having missing values
    data = data.dropna()
    
    # den Datentyp der Kundennummer gemäß dem Geschäftsverständnis ändern
    data['CustomerID'] = data['CustomerID'].astype(str)
    
    print("Shape after dropping rows having missing values", data.shape)
    
    # Objekttyp in datetime für InvoiceDate konvertieren und erstes und letztes Datum prüfen
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    print("Minimum Invoice Date", min(data["InvoiceDate"]))
    print("Maximum Invoice Date", max(data["InvoiceDate"]))
    
    # add cancellations column based on definition that InvoiceNo starts with C
    data["Cancel"] = np.where(data["InvoiceNo"].str.startswith('C'), 1,0)
    print(data["InvoiceNo"].shape[0])
    total_data = data["InvoiceNo"].shape[0]
    cancel_data = data[data.Cancel == 1].shape[0]
    print("Number of cancelled products:", cancel_data, np.ceil(cancel_data*100/total_data),"%", "\n")
    
    print(data[data.Cancel == 1]["Quantity"].describe())
    '''
    Dies zeigt, dass die Menge negativ ist, wenn eine Bestellung storniert wurde. 
    Stornierungsdaten wurden entfernt, da sie nur ~2% der Daten ausmachen.
    '''
    # Remove cancellations since they have negative quantities and makes only ~2.2% of data
    data = data[data.Cancel == 0]
    
    ##### check UnitPrice variable
    print("Minimum UnitPrice", min(data["UnitPrice"]))
    print("Maximum UnitPrice", max(data["UnitPrice"]))
    data["UnitPrice"].describe()
    # check where Price <0
    print(data[data.UnitPrice<0]) # No Price <0
    
    # check where Price = 0
    print("% of data with Price = 0: ", round(len(data[data.UnitPrice == 0]) * 100 / len(data),2),"%" )
    print("Count of unique Customer ID values ", data[data.UnitPrice == 0].CustomerID.nunique(), "\n")
    data[data.UnitPrice == 0][~data.CustomerID.isnull()].head()
    
    # remove  UnitPrice = 0
    data_clean = data[data.UnitPrice >0] # New data 
    print(data_clean["UnitPrice"].describe()) # min = 0.001
    
    print("Minimum Quantity", min(data_clean["Quantity"]))
    print("Maximum Quantity", max(data_clean["Quantity"]))
    print(data_clean["Quantity"].describe())
    
    ##### Aufteilung der Kunden und des Gesamtumsatzes nach Ländern
    
    ### Adding Monetary information column by calculating total value of transaction = unit price * quantity
    data_clean["TotalSales"] = data_clean["UnitPrice"]*data_clean["Quantity"]
    

    
    ### create country level groups to find unique customer count and %
    data_cc = data_clean.groupby("Country")["CustomerID"].nunique().reset_index().rename(columns = \
                                                                                   {"CustomerID":"count_CustomerID"})
    data_cc["Customer percentage"] = round(data_cc["count_CustomerID"]*\
                                          100/data_cc["count_CustomerID"].sum(),2)
        
        
     ### Creating Country Level grouping to find total revenue and %
    data_country = data_clean.groupby("Country")["TotalSales"].sum().reset_index()
    data_country["Total Sales%"] = round(data_country["TotalSales"]*100/data_country["TotalSales"].sum(),2)   
        
    data_cc = data_cc.sort_values(by = "Customer percentage", ascending = False)
    
    # plot customer count % vs. countries
    fig, ax = plt.subplots(figsize=(10,4),dpi=100)
    ax=sns.barplot(x=data_cc["Country"], y=data_cc['Customer percentage'],color='steelblue')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
    ax.set_title('Percentage of customers by countries', fontsize = 10)
    plt.show()
    
    # plot total sales % vs. countries
    data_country = data_country.sort_values(by = "Total Sales%", ascending = False)
    fig, ax = plt.subplots(figsize=(10,4),dpi=100)
    ax=sns.barplot(x=data_country["Country"], y=data_country['Total Sales%'],color='steelblue')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right")
    ax.set_title('Percentage of total sales by countries', fontsize = 10)
    plt.show()
    
    '''
    Das Vereinigte Königreich hat nicht nur die meisten Umsätze, sondern auch die meisten 
    Kunden. Daher werde ich für diese Analyse die Daten, 
    die sich auf Bestellungen aus dem Vereinigten Königreich beziehen.
    '''
    
    data_uk = data_clean[data_clean.Country == "United Kingdom"]
    print(data_uk.info())
    
    # checking most popular products in UK
    
    data_uk_prod = data_uk.groupby(['StockCode','Description'],as_index= False)['Quantity'].sum().sort_values(by='Quantity', \
                                                                                                      ascending=False)
    print(data_uk_prod.head(5))
    
    # find the unique number of InvoiceNo  per customer for analysis of United Kingdom 
    groupby_customer = pd.DataFrame(data_uk.groupby('CustomerID')['InvoiceNo'].nunique())
    groupby_customer.columns = ['Number of InvoiceNo  per Customer']
    print(groupby_customer.describe())
    
    # find the unique number of products per Order
    groupby_invoice = pd.DataFrame(data_uk.groupby('InvoiceNo')['StockCode'].nunique())
    groupby_invoice.columns = ['Number of products per Order']
    print(groupby_invoice.describe())
  
   
    ####  RFM Segmentation
 
    print('')
    print(" --------------------------- ")
    print(" 3. RFM data modeling ...  ")
    print(" -------------------------- ")
    
    date_ana = data_uk["InvoiceDate"].max() + pd.DateOffset(1) # analysis date
    print("RFM Analysis Date :", date_ana)
    
    start_date = date_ana - pd.DateOffset(days = 365)
    print("Start Date when taking 1 year data for analysis :", start_date)
  
    ### remove Nulls in CUstomerID
    data_uk = data_uk[~data_uk.CustomerID.isnull()]
    
    # aggregate data on a customer level to get RFM values
    rfmdata = data_uk[data_uk.InvoiceDate >= start_date].groupby(['CustomerID'],as_index=False).agg({'InvoiceDate': lambda x: (date_ana - x.max()).days,
    'InvoiceNo': 'count','TotalSales': 'sum'}).rename(columns = {'InvoiceDate': 'Recency', \
                                               'InvoiceNo': 'Frequency','TotalSales': 'Monetary'})
    print('Extract of the RFM Data:') 
    print(rfmdata.head(5))                                                         
   
        
    ### get individual RFM scores by using quantiles for each of the columns
    rfmdata['R_score'] = pd.qcut(rfmdata['Recency'], 4, labels=False)
    rfmdata['F_score'] = pd.qcut(rfmdata['Frequency'], 4, labels=False)
    rfmdata['M_score'] = pd.qcut(rfmdata['Monetary'], 4, labels=False)
    
    
    df_rfmscore = pd.DataFrame(rfmdata, columns=['R_score', 'F_score','M_score'])
    print(df_rfmscore.head(5))  
    
    ### Since a low Recency score means recent transactions and good customer, changine quantile values 
    ### so that low values rank highest ans vice versa
    rfmdata['R_score'] = 3 - rfmdata['R_score']
    
    rfmdata['RFM'] = rfmdata.R_score.map(str) \
                            + rfmdata.F_score.map(str) \
                            + rfmdata.M_score.map(str)

    ### Calculating Final RFM score
    rfmdata["RFM_Score"] = rfmdata['R_score'] + rfmdata['F_score'] + rfmdata['M_score']
    print(rfmdata.head(10))
    
   
    rfmmdata_mean = rfmdata.groupby("RFM_Score")[['Recency','Frequency', 'Monetary']].mean()
    print(rfmmdata_mean)
    
    print('')
    print(" ------------------------------ ")
    print(" 4. RFM data distribution ...  ")
    print(" ----------------------------- ")
    
    # Checking the distribution of Recency, Frequency and MonetaryValue variables.
    plt.figure(figsize=(12,10))
    # Plot distribution of Recency
    plt.subplot(3, 1, 1)
    sns.distplot(rfmdata['Recency'])
    # Plot distribution of Frequency
    plt.subplot(3, 1, 2)
    sns.distplot(rfmdata['Frequency'])
    # Plot distribution of Monetary
    plt.subplot(3, 1, 3) 
    sns.distplot(rfmdata['Monetary'])
    plt.show()
    
 
    print('')
    print(" ---------------------------------------- ")
    print(" 5. Log transformation of RFM data  ...  ")
    print(" ---------------------------------------- ")
    # Taking Log of columns
    rfmdata["log_Recency"] = np.log(rfmdata.Recency)
    rfmdata["log_Frequency"] = np.log(rfmdata.Frequency)
    rfmdata["log_Monetary"] = np.log(rfmdata.Monetary)
    
    # Checking the distribution of Recency, Frequency and Monetary variables after log transformation
    # Log Transformation Plots for R,F,M 
    plt.figure(figsize=(12,10))
    plt.subplot(3, 1, 1)
    sns.distplot(rfmdata['log_Recency'])
    plt.subplot(3, 1, 2)
    sns.distplot(rfmdata['log_Frequency'])
    plt.subplot(3, 1, 3)
    sns.distplot(rfmdata['log_Monetary'])
    plt.show()
    
    # save rfm data 
    rfmdata.to_csv('ordata_rfm.csv', index=False)
    
    return rfmdata
    

#%%
def perform_kmeans_ordata(data_rfm):
    
      print("")
      print(" ----------------------------------------------")
      print("  perform k-means on online retail data ...    ")
      print(" ----------------------------------------------") 
     
      # features Used in  K Means - Log Transformed Recency, Frequency and Monetary values
      data_norm_rfm = data_rfm[["log_Recency", "log_Frequency", "log_Monetary"]]

      print("")
      print(" ---------------------------------------")
      print(" 1. find optimal number of cluster ...  ")
      print(" -------------------------------------- ")
      print("")
      print(" a) with Elbow method using KElbowVisualizer ")
      # Elbow method using KElbowVisualizer
      plt.subplots(figsize=(12,8))
      kmeans = KMeans(random_state=1)
      visualizer = KElbowVisualizer(kmeans, k=(2,10), timings=False)
      visualizer.fit(data_norm_rfm)
      k_scores_elbow = visualizer.k_scores_  # Get elbow scores
      visualizer.show()
      plt.show()
      print("Elbow scores: ",k_scores_elbow)
      
      print("")
      print(" b) with silhouette score using KElbowVisualizer ")
      # Silhouette score
      plt.subplots(figsize=(12,8))
      kmeans = KMeans(random_state=1)
      visualizer = KElbowVisualizer(kmeans, k=(2,10), timings=False, metric='silhouette')
      visualizer.fit(data_norm_rfm)
      k_scores_silouhette = visualizer.k_scores_  # Get silhouette scores
      visualizer.show()
      plt.show()
      print("Silouhette scores: ",k_scores_silouhette)
      
      print("")
      print(" c) with Calinski-Harabasz score using KElbowVisualizer ")
      # Silhouette score
      plt.subplots(figsize=(12,8))
      kmeans = KMeans(random_state=1)
      visualizer = KElbowVisualizer(kmeans, k=(2,10), timings=False, metric='calinski_harabasz')
      visualizer.fit(data_norm_rfm)
      k_scores_ch = visualizer.k_scores_  # Get Calinski-Harabasz score
      visualizer.show()
      plt.show()
      print("Calinski-Harabasz scores: ",k_scores_ch)
      
      print("")
      print(" d) with SSE Method  ")
      ## SSE 
      
      sse = {}
    # Fit KMeans and calculate SSE for each k
      for k in range(1, 11):
        # Initialize KMeans with k clusters
        km = KMeans(n_clusters=k, random_state=1)
        # Fit KMeans on the normalized dataset
        km.fit(data_norm_rfm)
        # Assign sum of squared distances to k element of dictionary
        sse[k] = km.inertia_
        
    # Plotting the elbow plot
      plt.figure(figsize=(12,8))
      plt.title('The Elbow Method')
      plt.xlabel('k'); 
      plt.ylabel('Sum of squared errors')
      sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
      plt.show()
      print("   check plot SSE vs. k ")
      
    
      print("")
      print(" ---------------------------------------------------")
      print(" 2. perform clustering using k-Means with k = 3 ...  ")
      print(" --------------------------------------------------- ")
      print("")
      n_clusters = 3
      kmeans3 = KMeans(n_clusters = n_clusters, random_state=1)
      kmeans3.fit(data_norm_rfm)
      data_rfm["cluster"] = kmeans3.predict(data_norm_rfm)
      
      # Überprüfung der mittleren RFM-Werte in verschiedenen Clustern um die Clustereigenschaften zu verstehen
      
      print(" Mean RFM values in different clusters:  ")
      mean_clusters_rfm = data_rfm.groupby(["cluster"])[['Recency','Frequency', 'Monetary']].mean()
      print(mean_clusters_rfm)
      
      
      print("")
      print(" ------------------------------------------------------------")
      print(" 3. correlation between Recency, Frequency and Monetary ...  ")
      print(" ------------------------------------------------------------")
      print("")
      # Correlation matrix 
      plt.figure(figsize=(10,8))
      sns.heatmap(data_rfm[['Recency','Frequency', 'Monetary']].corr(),cmap="YlGnBu",annot=True)        
      plt.title('Correlation between Recency, Frequency and Monetary')
      plt.show()
    

      # define and map colors
      colors = ['g', 'r', 'b']
      data_rfm['color'] = data_rfm.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
    
      # Plot clusters
      prediction = data_rfm[['log_Frequency', 'log_Monetary','cluster']]
      clus0 = prediction.loc[prediction['cluster'] == 0]
      clus1 = prediction.loc[prediction['cluster']== 1]
      clus2 = prediction.loc[prediction['cluster']== 2]
     
      # remove cluster column
      clus0  = clus0.drop(columns=['cluster'])
      clus1  = clus1.drop(columns=['cluster'])
      clus2  = clus2.drop(columns=['cluster'])
      # DataFrame in Liste umwandeln 
      clus0 = clus0.values
      clus1 = clus1.values
      clus2 = clus2.values
      
      #print(clus0)
      # get centroids of each cluster
      
      print("")
      print(" ------------------------------------------------------------")
      print(" 4. Centers of 3 clusters  ...                               ")
      print(" ------------------------------------------------------------")
      print("")
      center0 = get_center(clus0)
      center1 = get_center(clus1)
      center2 = get_center(clus2)
      
      centers_x = [center0[0],center1[0],center2[0]] 
      centers_y = [center0[1],center1[1],center2[1]]
      print("Center of cluster 0: ",(center0[0],center0[1])) 
      print("Center of cluster 1: ",(center1[0],center1[1]))
      print("Center of cluster 2: ",(center2[0],center2[1]))
      
      # log frequency and log m onetary of cluster 0 
      clus0_freq = prediction[prediction['cluster'] == 0]['log_Frequency'].values
      clus0_monet = prediction[prediction['cluster'] == 0]['log_Monetary'].values
      # log frequency and log m onetary of cluster 1 
      clus1_freq = prediction[prediction['cluster'] == 1]['log_Frequency'].values
      clus1_monet = prediction[prediction['cluster'] == 1]['log_Monetary'].values
      # log frequency and log m onetary of cluster 2
      clus2_freq = prediction[prediction['cluster'] == 2]['log_Frequency'].values
      clus2_monet = prediction[prediction['cluster'] == 2]['log_Monetary'].values
    
      print("")
      print(" ------------------------------------------------------------")
      print(" 5. plot cluster incl. their centers ...                               ")
      print(" ------------------------------------------------------------")
      print("")
      # plot clusters and their centers
      plt.figure(figsize=(6,6))
      plt.scatter(clus0_freq, clus0_monet, c = 'g', label = 'Cluster 0', s=50)
      plt.scatter(clus1_freq, clus1_monet, c = 'r', label = 'Cluster 1', s=50)
      plt.scatter(clus2_freq, clus2_monet, c = 'b', label = 'Cluster 2', s=50)
      plt.scatter(centers_x, centers_y, s = 50, c = 'black', label = 'Centroids')
      plt.title('Clustering of online retail data and their centers using K-Means')
      plt.xlabel('Log Frequency', fontsize=15)
      plt.ylabel('Log Monetary', fontsize=15)
      plt.legend()
      plt.show()
      
      print("")
      print(" -----------------------------------------------------")
      print(" 6. compute cluster  sizes ...                        ")
      print(" ----------------------------------------------------- ")  

      # eine kmeans-Instanz mit berechnetem k erstellen
      k_means = kmeans3.fit(data_norm_rfm) # K-Means-Modell initialisieren und anpassen
      data_norm_rfm = pd.DataFrame(data_norm_rfm,columns = ['log_Frequency','log_Monetary'])
      data_norm_rfm.loc[:,'Cluster Nr.'] = k_means.labels_ # append labels to points
      cluster_sizes = data_norm_rfm.groupby('Cluster Nr.').size().to_frame()
      cluster_sizes.columns = ["Cluster Size"]
      
      # displaying the DataFrame
      print(tabulate(cluster_sizes, headers = 'keys', tablefmt = "fancy_grid"))
      

#%%

def perform_dbscan_ordata(data_rfm):
    
      print("")
      print(" ----------------------------------------------")
      print("  perform dbscan on online retail data ...      ")
      print(" ----------------------------------------------")  
      print("")
      print(" -----------------------------------------------------")
      print(" 1. find optimal parameters (eps,min-samples) ...     ")
      print(" ----------------------------------------------------- ")  
      
      # features used: Log Transformed Recency, Frequency and Monetary values
      data_norm_rfm = data_rfm[["log_Recency", "log_Frequency", "log_Monetary"]]
      #data_norm_rfm = data_rfm[["log_Frequency", "log_Monetary"]]
      
  

######     
      # das passende Epsilon mit der „Elbow-Methode“ bestimmen 
      plt.figure(figsize=(10,5))
      nn = NearestNeighbors(n_neighbors=5).fit(data_norm_rfm)
      distances, idx = nn.kneighbors(data_norm_rfm)
      distances = np.sort(distances, axis=0)
      distances = distances[:,1]
      plt.plot(distances)
      plt.show()
        
      # nach entsprechenden min_samples suchen
      eps_values = np.arange(0.2,1.5,0.1) # eps values to be investigated
      min_samples = np.arange(2,5) # min_samples values to be investigated
      dbscan_params = list(product(eps_values, min_samples))
      no_of_clusters = []
      sil_scores = []
      epsvalues = []
      min_samp = []
      eps_minsamples = []
      for p in dbscan_params:
                dbscan_cluster = DBSCAN(eps=p[0], min_samples=p[1]).fit(data_norm_rfm)
                epsvalues.append(p[0])
                min_samp.append(p[1])
                no_of_clusters.append(len(np.unique(dbscan_cluster.labels_)))
                sil_scores.append(silhouette_score(data_norm_rfm, dbscan_cluster.labels_))
                eps_minsamples.append(p)
              
      eps_min = list(zip(no_of_clusters, sil_scores, epsvalues, min_samp))
      eps_min_df = pd.DataFrame(eps_min, columns=['Nr. clusters', 'Silhouette score', 'eps', 'minPts'])
      print(tabulate(eps_min_df, headers = 'keys', tablefmt = 'fancy_grid',showindex=False))
              
     
      df1 = pd.DataFrame.from_records(dbscan_params, columns =['epsilon', 'min_samples'])   
      df1['No_of_clusters'] = no_of_clusters
    
      pivot_1 = pd.pivot_table(df1, values='No_of_clusters', index='min_samples', columns='epsilon')
    
      fig1, ax1 = plt.subplots(figsize=(14,4))
      sns.heatmap(pivot_1, annot=True,annot_kws={"size": 10}, cmap="Blues", ax=ax1)
      ax1.set_title('Number of clusters for DBSCAN')
      plt.show()
      
      # find best combination of eps and min_samples
      index_max_sil = sil_scores.index(max(sil_scores))
      print("Max. silouhette score: ",max(sil_scores))
      print("Number of clusters: ",no_of_clusters[index_max_sil])
      print("Optimal parameters (eps,min_samples): ",eps_minsamples[index_max_sil])
      
      df2= pd.DataFrame.from_records(dbscan_params, columns =['epsilon', 'min_samples'])   
      df2['Sil_Scores'] = np.floor(sil_scores)
      pivot_2 = pd.pivot_table(df2, values='Sil_Scores', index='min_samples', columns='epsilon')
      fig2, ax2 = plt.subplots(figsize=(12,6))
      sns.heatmap(pivot_2, annot=True, annot_kws={"size": 10}, cmap="YlGnBu", ax=ax2)
      ax2.set_title('Silouhette Scores using DBSCAN')
      plt.show()
      
      # berechnete optimale eps und min_samples festlegen
      (epsilon,minsamples) = eps_minsamples[index_max_sil]
      
     ######  Eps und minPts festlegen 

      epsilon = 1.3 #1.4000000000000004 #1.3 #0.7
      minsamples = 2# 4#2 #3
      
      print("")
      print(" --------------------------------------------------------------------------")
      print(" 2. apply DBSCAN using optimal parameters eps = "+ str(epsilon)+ " and min_samples = "+ str(minsamples))
      print(" --------------------------------------------------------------------------")  
      
      dbscan = DBSCAN(eps=epsilon, min_samples=minsamples).fit(data_norm_rfm)
      dbscan_ = data_norm_rfm.copy()
      dbscan_.loc[:,'Cluster Nr.'] = dbscan.labels_ # append labels to points
        
      #check sizes of clusters.
      print("")
      print(" ---------------------------------------------------------------")
      print(" 3. compute cluster sizes ... ")
      print(" --------------------------------------------------------------- ")  
     
      dbscan_cls = dbscan_.groupby('Cluster Nr.').size().to_frame()
      dbscan_cls.columns = ["Cluster Size"]
      print(tabulate(dbscan_cls, headers = 'keys', tablefmt = "fancy_grid"))
   
    
      dbscan_.loc[:,'Cluster'] = dbscan.labels_  # append labels to points
      data_rfm["cluster"] = dbscan.fit_predict(data_norm_rfm)
      outliers = dbscan_[dbscan_['Cluster']==-1]
      
      #define and map colors
      colors = ['g', 'r', 'b']
      data_rfm['color'] = data_rfm.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
    
      # Plot clusters
      prediction = data_rfm[['log_Frequency', 'log_Monetary','cluster']]
      clus0 = prediction.loc[prediction['cluster'] == 0]
      clus1 = prediction.loc[prediction['cluster']== 1]
      clus2 = prediction.loc[prediction['cluster']== 2]
    
      # remove cluster column
      clus0  = clus0.drop(columns=['cluster'])
      clus1  = clus1.drop(columns=['cluster'])
      clus2  = clus2.drop(columns=['cluster'])
      # DataFrame in Liste umwandeln 
      clus0 = clus0.values
      clus1 = clus1.values
      clus2 = clus2.values
      
      print("")
      print(" ------------------------------------------------------------")
      print(" 4. get centers of clusters  ...                               ")
      print(" ------------------------------------------------------------")
      print("")
      # get centroids of each cluster
      center0 = get_center(clus0)
      center1 = get_center(clus1)
      center2 = get_center(clus2)
      
      centers_x = [center0[0],center1[0],center2[0]] 
      centers_y = [center0[1],center1[1],center2[1]]
      print("Center of cluster 0: ",(center0[0],center0[1])) 
      print("Center of cluster 1: ",(center1[0],center1[1]))
      print("Center of cluster 2: ",(center2[0],center2[1]))
      
      # log frequency and log m onetary of cluster 0 
      clus0_freq = prediction[prediction['cluster'] == 0]['log_Frequency'].values
      clus0_monet = prediction[prediction['cluster'] == 0]['log_Monetary'].values
      # log frequency and log m onetary of cluster 1 
      clus1_freq = prediction[prediction['cluster'] == 1]['log_Frequency'].values
      clus1_monet = prediction[prediction['cluster'] == 1]['log_Monetary'].values
      # log frequency and log m onetary of cluster 2
      clus2_freq = prediction[prediction['cluster'] == 2]['log_Frequency'].values
      clus2_monet = prediction[prediction['cluster'] == 2]['log_Monetary'].values
      
      print("")
      print(" ------------------------------------------------------------")
      print(" 5. plot clusters incl. their centers ...                               ")
      print(" ------------------------------------------------------------")
      print("")
      # plot clusters and their centers
      plt.figure(figsize=(6,6))
      plt.scatter(clus0_freq, clus0_monet, c = 'g', label = 'Cluster 0', s=50)
      plt.scatter(clus1_freq, clus1_monet, c = 'r', label = 'Cluster 1', s=50)
      plt.scatter(clus2_freq, clus2_monet, c = 'b', label = 'Cluster 2', s=50)
      plt.scatter(centers_x, centers_y, s = 60, c = 'm', label = 'Centroids')
      plt.scatter(x = outliers['log_Frequency'], y = outliers['log_Monetary'], s=100, label='outliers', c="k")
      plt.title('Clustering of online retail data and their centers using DBSCAN')
      plt.xlabel('Log Frequency', fontsize=15)
      plt.ylabel('Log Monetary', fontsize=15)
      plt.legend()
      plt.show()
      
      print("")
      print(" --------------------------")
      print(" 6. show outliers data ... ")
      print(" ------------------------- ")  
      outliers  = outliers.drop(columns=['Cluster'])
      print(tabulate(outliers, headers = 'keys', tablefmt = 'fancy_grid',showindex=False))
         
      
#%%
def compare_performance_kmeans_dbscan_ordata(data_rfm): 
    
    print("")
    print(" --------------------------------------------------------------------------")
    print("  Clustering Evaluation for online retail data using k-means and dbscan ...  ")
    print(" --------------------------------------------------------------------------")  
    print("")
    data_norm_rfm = data_rfm[["log_Recency", "log_Frequency", "log_Monetary"]]
    ### K-Means
    print(" --------------------------")
    print(" 1) K-Means evaluation ... ")
    print(" ------------------------- ")  
    km_times = []
    for i in range(0,10):
        start_km = timer() # start timer
        k_means = KMeans(n_clusters=3, init='k-means++',max_iter=300,n_init=10,random_state=0).fit(data_norm_rfm) 
        end_km = timer() # end timer
        km_times.append(end_km - start_km)
    km_t = np.mean(km_times)
    
    # Gültigkeits-Clustering-Indizes für k-Means berechnen
    cluster_labels_km = k_means.fit_predict(data_norm_rfm)
    silhouette_km = silhouette_score(data_norm_rfm, cluster_labels_km)
    davies_bouldin_km = davies_bouldin_score(data_norm_rfm, cluster_labels_km)
    calinski_harabasz_km = calinski_harabasz_score(data_norm_rfm, cluster_labels_km)
    
    print("  done!...  ")
    
    
    # epsilon =1.3 #0.7
    # minsamples = 2 #3
   
    ### DBSCAN 
    print("")
    print(" --------------------------")
    print(" 2) DBSCAN evaluation ...  ")
    print(" ------------------------- ")  
    dbs_times = []
    for i in range(0,10):
        start_ds= timer() # start timer
        dbscan = DBSCAN(eps=1.3, min_samples=2).fit(data_norm_rfm)
        end_ds = timer() # end timer
        dbs_times.append(end_ds - start_ds)
    dbs_t = np.mean(dbs_times)
    # Gültigkeits-Clustering-Indizes für DBSCAN berechnen
    cluster_labels_ds = dbscan.fit_predict(data_norm_rfm)
    silhouette_ds = silhouette_score(data_norm_rfm, cluster_labels_ds)
    davies_bouldin_ds = davies_bouldin_score(data_norm_rfm, cluster_labels_ds)
    calinski_harabasz_ds = calinski_harabasz_score(data_norm_rfm, cluster_labels_ds)
    print("  done!...  ")
    
    print("")
    print(" --------------------------")
    print(" 3) Results ...            ")
    print(" ------------------------- ")  
    
    # Ergebnisse in DataFrame umwandeln 
    dict = {'Metric':['TIME (s)','SI','DB', 'CH'],
        'K-Means':[km_t,silhouette_km, davies_bouldin_km, calinski_harabasz_km],
        'DBSCAN':[dbs_t,silhouette_ds, davies_bouldin_ds, calinski_harabasz_ds]} 
    df_results = pd.DataFrame(dict)
    
    print(tabulate(df_results, headers = 'keys', tablefmt = 'fancy_grid',showindex=False))

        
#%%
if __name__ == "__main__":
        
    ordata = 'onlineRetail.csv' # original data
    
    '''Data chek and preparation'''
    data_rfm = check_prepare_ordata(ordata) 
    
    '''Perform K-Means'''
    perform_kmeans_ordata(data_rfm)
  
    '''Perform DBSCAN'''
    perform_dbscan_ordata(data_rfm)
  
    '''Compare K-Means with DBSCAN'''
    compare_performance_kmeans_dbscan_ordata(data_rfm)

           
        
        
        

        
