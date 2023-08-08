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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator
from yellowbrick.cluster import KElbowVisualizer
from tabulate import tabulate
from itertools import product
from timeit import default_timer as timer
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings


# %%  Implemented functions
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
    center = cluster_array.mean(axis=0)
    return center


# %%
def check_prepare_malldata(csv_data):
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

    data = pd.read_csv(csv_data)

    # Data infos
    print('Data shape: There are {} rows and {} columns '.format(data.shape[0], data.shape[1]))
    print("")
    print('Columns names:', list(data.columns.values.tolist()))
    print("")
    print("First five samples: ")
    print(data.head())
    print("")
    print("Summary statistics of variables: ")
    print(data.describe())
    print("")
    print("Check missing data: ")  
    if (data.isnull().sum().sum() == 0):
        print("No missing data as shown here:")
        print(data.isnull().sum())

    else:
        print("There are missing data.")
        print(data.isnull().sum())
        print("Missing data are in the following columns:")
        print(data.columns[data.isna().any()])

    #print(data.shape)
    #print(data.info())
    #print(data.columns)
    #print(data.head())
    #print(data.isna().sum())
    #print(data.isnull().sum())
    
    print("")
    print(" --------------------------------")
    print(" 2. check data distribution ...  ")
    print(" -------------------------------- ")
    print(" see plots and histograms ...  ")

    # Gender distribution
    plt.figure(figsize=(8, 8))
    sns.countplot(x='Gender', data=data, palette=['b', "g"])
    plt.title("Gender distribution")
    #plt.legend()
    plt.show()

    # ----- histogram age by gender

    # Extract age for males
    age_male = data[data['Gender'] == 'Male']['Age']  # subset with males age

    # Extract ages for females
    # subset with females age
    age_female = data[data['Gender'] == 'Female']['Age']

    # Plot histogram age by gender
    age_bins = range(15, 75, 5)

    # male histogram
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig1.suptitle("Histogram Age by Gender", fontsize="x-large")
    sns.distplot(age_male, bins=age_bins, kde=False, color='b',
                 ax=ax1, hist_kws=dict(edgecolor="w", linewidth=2))
    ax1.set_xticks(age_bins)
    ax1.set_ylim(top=25)
    ax1.set_title('Male')
    ax1.set_ylabel('Count')
    ax1.text(45, 23, "Total: {}".format(age_male.count()))
    ax1.text(45, 22, "Mean: {:.1f}".format(age_male.mean()))

    # female histogram
    sns.distplot(age_female, bins=age_bins, kde=False, color='g',
                 ax=ax2, hist_kws=dict(edgecolor="w", linewidth=2))
    ax2.set_xticks(age_bins)
    ax2.set_title('Female')
    ax2.set_ylabel('Count')
    ax2.text(45, 23, "Total: {}".format(age_female.count()))
    ax2.text(45, 22, "Mean: {:.1f}".format(age_female.mean()))
    plt.show()

    # ----- histogram income by gender

    # Extract income for males
    income_male = data[data['Gender'] =='Male']['Annual Income (k$)']  # subset with males age

    # Extract income for females
    # subset with females age
    income_female = data[data['Gender'] == 'Female']['Annual Income (k$)']

    # Plot histogram income by gender
    income_bins = range(10, 150, 15)

    # male histogram
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig2.suptitle("Histogram Income by Gender", fontsize="x-large")
    sns.distplot(income_male, bins=income_bins, kde=False, color='b',
                 ax=ax3, hist_kws=dict(edgecolor="w", linewidth=2))
    ax3.set_xticks(income_bins)
    ax3.set_ylim(top=25)
    ax3.set_title('Male')
    ax3.set_ylabel('Count')
    ax3.text(85, 23, "Total: {}".format(income_male.count()))
    ax3.text(85, 22, "Mean: {:.1f}".format(income_male.mean()))

    # female histogram
    sns.distplot(income_female, bins=income_bins, kde=False,
                 color='g', ax=ax4, hist_kws=dict(edgecolor="w", linewidth=2))
    ax4.set_xticks(income_bins)
    ax4.set_title('Female')
    ax4.set_ylabel('Count')
    ax4.text(85, 23, "Total: {}".format(income_female.count()))
    ax4.text(85, 22, "Mean: {:.1f}".format(income_female.mean()))
    plt.show()

    # ----- histogram Spending Score (1-100) by gender

    # Extract spending for males
    # subset with males age
    spending_male = data[data['Gender'] == 'Male']['Spending Score (1-100)']

    # Extract spending for females
    # subset with females age
    spending_female = data[data['Gender'] ==
                           'Female']['Spending Score (1-100)']

    # Plot histogram spending by gender
    spending_bins = range(0, 100, 10)

    # male histogram
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig3.suptitle("Histogram Spending by Gender", fontsize="x-large")
    sns.distplot(spending_male, bins=spending_bins, kde=False,
                 color='b', ax=ax5, hist_kws=dict(edgecolor="w", linewidth=2))
    ax5.set_xticks(spending_bins)
    ax5.set_ylim(top=30)
    ax5.set_title('Male')
    ax5.set_ylabel('Count')
    ax5.text(45, 23, "Total: {}".format(spending_male.count()))
    ax5.text(45, 22, "Mean: {:.1f}".format(spending_male.mean()))

    # female histogram
    sns.distplot(spending_female, bins=spending_bins, kde=False,
                 color='g', ax=ax6, hist_kws=dict(edgecolor="w", linewidth=2))
    ax6.set_xticks(spending_bins)
    ax6.set_title('Female')
    ax6.set_ylabel('Count')
    ax6.text(50, 23, "Total: {}".format(spending_female.count()))
    ax6.text(50, 22, "Mean: {:.1f}".format(spending_female.mean()))
    plt.show()

    # Median annual income of male and female
    med_by_age_group = data.groupby(
        ["Gender", pd.cut(data['Age'], age_bins)]).median()
    med_by_age_group.index = med_by_age_group.index.set_names(
        ['Gender', 'Age_group'])
    med_by_age_group.reset_index(inplace=True)

    fig4, ax7 = plt.subplots(figsize=(12, 5))
    sns.barplot(x='Age_group', y='Annual Income (k$)', hue='Gender', data=med_by_age_group,
                palette=['b', 'g'],
                alpha=0.7, edgecolor='w',
                ax=ax7)
    ax7.set_title('Histogram: Median income of male vs. female')
    ax7.set_xlabel('Age')
    plt.show()

    print("")
    print(" ------------------------------")
    print(" 3. check data correlation ...  ")
    print(" ------------------------------ ")

    data = data.drop('CustomerID', axis=1)  # drop customer IDs

    # Correlation matrix
    corr = data.corr()
    print("Correlation Matrix:\n", corr) #--------------------------
    sns.heatmap(corr, cmap="YlGnBu", annot=True)  
    plt.title('Correlation matrix')
    plt.show()

    # displaying the DataFrame
    print(" correlation matrix: ")
    print(tabulate(corr, headers='keys', tablefmt="fancy_grid"))

    sns.pairplot(data, palette='Set1')
    plt.show()

    col_names = data.select_dtypes(include=np.number).columns.tolist()
    plt.figure(1, figsize=(15, 10))
    i = 0
    for s in col_names:
        for r in col_names:
            i += 1
            plt.subplot(4, 4, i)
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            sns.regplot(x=s, y=r, data=data, scatter_kws={
                        "color": "blue"}, line_kws={"color": "green"})
            plt.ylabel(r.split()[0]+''+r.split()[1]
                       if len(r.split()) > 1 else r)
    #plt.title('Correlation between variables')
    plt.show()

    # Annual Income vs Spending Score with Gender
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='Spending Score (1-100)',
                    y='Annual Income (k$)', hue='Gender', data=data)
    plt.title("Income vs. Spending")
    plt.show()

    # Age vs Spending score with Gender
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='Spending Score (1-100)',
                    y='Age', hue="Gender", data=data)
    plt.title("Age vs Spending score")
    plt.show()
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='Age',
                    y='Spending Score (1-100)', hue="Gender", data=data)
    plt.title("Age vs Spending score")
    plt.show()


# %%
def perform_kmeans_malldata(csv_data):

    print("")
    print(" ----------------------------------------------")
    print("  perform k-means on mall data ...             ")
    print(" ----------------------------------------------")
    data = pd.read_csv(csv_data)  # Read data from csv-file

    print("")
    print(" ---------------------------------------")
    print(" 1. find optimal number of cluster ...  ")
    print(" -------------------------------------- ")

    col_income_spending = ['Annual Income (k$)', 'Spending Score (1-100)']

    income_spending = data[col_income_spending]

    for col in col_income_spending:
     # fit on training data column
        scale = StandardScaler().fit(income_spending[[col]])

     # transform the training data column
        income_spending[col] = scale.transform(income_spending[[col]])

    print("")
    print(" a) with Elbow method using KElbowVisualizer ")
    # Elbow method using KElbowVisualizer
    plt.subplots(figsize=(12, 8))
    kmeans = KMeans(random_state=1)  
    visualizer = KElbowVisualizer(kmeans, k=(2, 10), timings=False)
    visualizer.fit(income_spending)
    k_scores_elbow = visualizer.k_scores_  # Get elbow scores
    visualizer.show()
    plt.show()
    print("Elbow scores: ", k_scores_elbow)

    print("")
    print(" b) with silhouette method using KElbowVisualizer ")
    # silhouette score
    plt.subplots(figsize=(12, 8))
    kmeans = KMeans(random_state=1)
    visualizer = KElbowVisualizer(kmeans, k=(2, 10), timings=False, metric='silhouette')
    visualizer.fit(income_spending)

    k_scores_silouhette = visualizer.k_scores_  # Get silhouette scores
    visualizer.show()
    plt.show()
    print("Silouhette scores: ", k_scores_silouhette)

    income_score = data.iloc[:, [False, False, False, True, True]].values 
    scaler = MinMaxScaler()
    income_score_scale = scaler.fit_transform(income_score)
    # KneeLocator
    sse = []
    K = range(2, 10)
    for k in K:
        kmean = KMeans(n_clusters=k)
        kmean.fit(income_score_scale)
        sse.append(kmean.inertia_)

    # Get the knee point
    kn = KneeLocator(x=K, y=sse, curve='convex', direction='decreasing')
    knee_point = kn.knee

    print("")
    print(" -----------------------------------------------------")
    print(" 2. apply k-means using optimal number of clusters = "+str(knee_point))
    print(" ----------------------------------------------------- ")
    # Create kmeans with optimal k
    kmeans = KMeans(n_clusters=knee_point, init='k-means++', max_iter=300, n_init=10, random_state=0)

    y_kmeans = kmeans.fit_predict(income_score)

    print("")
    print(" -----------------------------------------------------")
    print(" 3. draw clusters incl. their centers ...             ")
    print(" ----------------------------------------------------- ")

    # Plot clusters 0, 1,2,3 and 4 and their centers
    plt.figure(figsize=(6, 6))
    plt.scatter(income_score[y_kmeans == 0, 0],
                income_score[y_kmeans == 0, 1], s=45, c='g', label='Cluster 0')
    plt.scatter(income_score[y_kmeans == 1, 0],
                income_score[y_kmeans == 1, 1], s=45, c='b', label='Cluster 1')
    plt.scatter(income_score[y_kmeans == 2, 0],
                income_score[y_kmeans == 2, 1], s=45, c='r', label='Cluster 2')
    plt.scatter(income_score[y_kmeans == 3, 0], income_score[y_kmeans ==
                3, 1], s=45, c='burlywood', label='Cluster 3')
    plt.scatter(income_score[y_kmeans == 4, 0],
                income_score[y_kmeans == 4, 1], s=45, c='green', label='Cluster 4')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                :, 1], s=45, c='black', label='Centroids')
    plt.title('Clusters and their centers for K-Means')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()

    print("")
    print(" -----------------------------------------------------")
    print(" 4. compute cluster  sizes ...                        ")
    print(" ----------------------------------------------------- ")

    # eine kmeans-Instanz mit berechnetem k erstellen
    k_means = kmeans.fit(income_spending)  # K-Means-Modell initialisieren und anpassen
    income_spending_ = income_spending.copy()
    # append labels to points
    income_spending_.loc[:, 'Cluster Nr.'] = k_means.labels_
    cluster_sizes = income_spending_.groupby('Cluster Nr.').size().to_frame()
    cluster_sizes.columns = ["Cluster Size"]

    # displaying the DataFrame
    print(tabulate(cluster_sizes, headers='keys', tablefmt="fancy_grid"))


# %%
def perform_dbscan_malldata(csv_data):
    print("")
    print(" ----------------------------------------------")
    print("  perform dbscan on mall data ...             ")
    print(" ----------------------------------------------")

    data = pd.read_csv(csv_data)  # Read data from csv-file
    # Teilmenge nur mit numerischen Variablen
    data = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    print("")
    print(" -----------------------------------------------------")
    print(" 1. find optimal parameters (eps,min-samples) ...     ")
    print(" ----------------------------------------------------- ")

    epsilon = np.arange(8, 12.75, 0.25)  # eps values to be investigated
    minPts = np.arange(3, 10)  # min_samples values to be investigated

    params = list(product(epsilon, minPts))
    # Weil DBSCAN anhand dieser beiden Parameter selbst Cluster erstellt
    # Prüfen wir die Anzahl der erzeugten Cluster.
    nr_clusters = []
    sil_scores = []
    eps_minsamples = []

    for p in params:
        dbscan = DBSCAN(eps=p[0], min_samples=p[1]).fit(data)
        nr_clusters.append(len(np.unique(dbscan.labels_)))
        sil_scores.append(silhouette_score(data, dbscan.labels_))
        eps_minsamples.append(p)

    df1 = pd.DataFrame.from_records(params, columns=['Eps', 'Min_samples'])
    df1['No_of_clusters'] = nr_clusters

    pivot_1 = pd.pivot_table(df1, values='No_of_clusters',
                             index='Min_samples', columns='Eps')
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_1, annot=True, annot_kws={
                "size": 16}, cmap="Blues", ax=ax1)
    ax1.set_title('Number of clustersfor DBSCAN')
    plt.show()

    # find best combination of eps and min_samples
    index_max_sil = sil_scores.index(max(sil_scores))
    print("Max. silouhette score: ", max(sil_scores))
    print("Number of clusters: ", nr_clusters[index_max_sil])
    print("Optimal parameters (eps,min_samples): ",
          eps_minsamples[index_max_sil])

    df2 = pd.DataFrame.from_records(params, columns=['Eps', 'Min_samples'])
    df2['Sil_Scores'] = sil_scores
    pivot_1 = pd.pivot_table(df2, values='Sil_Scores',
                             index='Min_samples', columns='Eps')
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_1, annot=True, annot_kws={
                "size": 10}, cmap="YlGnBu", ax=ax2)
    ax2.set_title('Silouhette Scores for DBSCAN')
    plt.show()

    (epsilon, minsamples) = eps_minsamples[index_max_sil]
    print("")
    print(" --------------------------------------------------------------------------")
    print(" 2. apply DBSCAN using optimal parameters eps = " +
          str(epsilon) + " and min_samples = " + str(minsamples))
    print(" --------------------------------------------------------------------------")

    dbscan = DBSCAN(eps=epsilon, min_samples=minsamples).fit(data)
    dbscan_ = data.copy()
    dbscan_.loc[:, 'Cluster Nr.'] = dbscan.labels_  # append labels to points

    # Checking sizes of clusters. 
    print("")
    print(" ---------------------------------------------------------------")
    print(" 3. compute cluster sizes ... ")
    print(" --------------------------------------------------------------- ")

    dbscan_cls = dbscan_.groupby('Cluster Nr.').size().to_frame()
    dbscan_cls.columns = ["Cluster Size"]
    # displaying the DataFrame
    print(tabulate(dbscan_cls, headers='keys', tablefmt="fancy_grid"))

    dbscan_ = data.copy()
    dbscan_.loc[:, 'Cluster'] = dbscan.labels_  # append labels to points

    outliers = dbscan_[dbscan_['Cluster'] == -1]
    y_pred = dbscan.fit_predict(data)



    # Wir speichern die DBSCAN-Ergebnisse in DataFrame           
    pred = pd.DataFrame(y_pred)
    pred.columns = ['cluster']
    prediction = pd.concat([data, pred], axis=1)

    clus0 = prediction.loc[prediction['cluster'] == 0]
    clus1 = prediction.loc[prediction['cluster'] == 1]
    clus2 = prediction.loc[prediction['cluster'] == 2]
    clus3 = prediction.loc[prediction['cluster'] == 3]
    clus4 = prediction.loc[prediction['cluster'] == 4]
                                               
    clus0_income_spend = clus0.drop(columns=['Age', 'cluster'])
    clus1_income_spend = clus1.drop(columns=['Age', 'cluster'])
    clus2_income_spend = clus2.drop(columns=['Age', 'cluster'])
    clus3_income_spend = clus3.drop(columns=['Age', 'cluster'])
    clus4_income_spend = clus4.drop(columns=['Age', 'cluster'])

    # der Zentren jedes Clusters ermitteln
    center0 = get_center(clus0_income_spend.values)
    center1 = get_center(clus1_income_spend.values)
    center2 = get_center(clus2_income_spend.values)
    center3 = get_center(clus3_income_spend.values)
    center4 = get_center(clus4_income_spend.values)
    centers_x = [center0[0], center1[0], center2[0], center3[0], center4[0]]
    centers_y = [center0[1], center1[1], center2[1], center3[1], center4[1]]

    print("")
    print(" ---------------------------------------------------------------")
    print(" 4. Draw clusters incl. outliers and centers ... ")
    print(" --------------------------------------------------------------- ")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                    data=dbscan_[dbscan_['Cluster'] != -1],
                    hue='Cluster', ax=ax, palette='Set1', legend='full', s=45)
    ax.scatter(x=outliers['Annual Income (k$)'],
               y=outliers['Spending Score (1-100)'], s=45, label='outliers', c="k")
    ax.scatter(x=centers_x, y=centers_y, s=100, c='m', label='Centroids')
    ax.legend()
    plt.setp(ax.get_legend().get_texts(), fontsize='10')
    plt.title('Clusters of Mall Customers using DBSCAN incl. Outliers and Centers.')
    plt.show()

    print("")
    print(" --------------------------")
    print(" 5. Show outliers data ... ")
    print(" ------------------------- ")
    print(tabulate(outliers, headers='keys', tablefmt='fancy_grid', showindex=False))


# %%
def compare_performance_kmeans_dbscan_malldata(csv_data):

    print("")
    print(" --------------------------------------------------------------------------")
    print("  Clustering Evaluation for mall data using k-means and dbscan ...       ")
    print(" -------------------------------------------------------------------------- ")

    data = pd.read_csv(csv_data)  # Read data from csv-file
    # Teilmenge nur mit numerischen Variablen
    data = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    print("")
    print(" --------------------------")
    print(" 1) K-Means evaluation ... ")
    print(" ------------------------- ")

    # K-Means
    km_times = []
    for i in range(0, 10):
        start_km = timer()  # start timer
        k_means = KMeans(n_clusters=5, init='k-means++',
                         max_iter=300, n_init=10, random_state=0).fit(data)
        end_km = timer()  # end timer
        km_times.append(end_km - start_km)
    km_t = np.mean(km_times)

    # Gültigkeits-Clustering-Indizes für k-Means berechnen
    cluster_labels_km = k_means.fit_predict(data)
    silhouette_km = silhouette_score(data, cluster_labels_km)
    davies_bouldin_km = davies_bouldin_score(data, cluster_labels_km)
    calinski_harabasz_km = calinski_harabasz_score(data, cluster_labels_km)
    print(" done! ... ")

    print("")
    print(" --------------------------")
    print(" 2) DBSCAN evaluation ... ")
    print(" ------------------------- ")

    # DBSCAN
    dbs_times = []
    for i in range(0, 10):
        start_ds = timer()  # start timer
        dbscan = DBSCAN(eps=12.5, min_samples=4).fit(data)
        end_ds = timer()  # end timer
        dbs_times.append(end_ds - start_ds)
    dbs_t = np.mean(dbs_times)

    # Gültigkeits-Clustering-Indizes für DBSCAN berechnen
    cluster_labels_ds = dbscan.fit_predict(data)
    silhouette_ds = silhouette_score(data, cluster_labels_ds)
    davies_bouldin_ds = davies_bouldin_score(data, cluster_labels_ds)
    calinski_harabasz_ds = calinski_harabasz_score(data, cluster_labels_ds)
    print(" done! ... ")

    print("")
    print(" --------------------------")
    print(" 3) Results ...            ")
    print(" ------------------------- ")

    # Ergebnisse in DataFrame umwandeln
    dict = {'Metric': ['TIME (s)', 'SI', 'DB', 'CH'],  # 'DI'],
            'K-Means': [km_t, silhouette_km, davies_bouldin_km, calinski_harabasz_km],
            'DBSCAN': [dbs_t, silhouette_ds, davies_bouldin_ds, calinski_harabasz_ds]}
    df_results = pd.DataFrame(dict)

    print(tabulate(df_results, headers='keys', tablefmt='fancy_grid', showindex=False))





# %%
if __name__ == "__main__":
    mall = "Mall_Customers.csv"
    
    '''Data chek and preparation'''
    check_prepare_malldata(mall)
    
    '''Perform K-Means'''
    perform_kmeans_malldata(mall)
    
    '''Perform DBSCAN'''
    perform_dbscan_malldata(mall)
    
    '''Compare K-Means with DBSCAN'''
    compare_performance_kmeans_dbscan_malldata(mall)
