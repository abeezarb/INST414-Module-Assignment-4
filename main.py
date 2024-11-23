import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('Expanded_data_with_more_features.csv')
df.dropna(inplace=True)

features = df[['MathScore','ReadingScore','WritingScore']]
features.head()

testScores = list()
for k in range(1,11):
    model = KMeans(n_clusters=k)
    model.fit(features)
    inertia_score = model.inertia_
    testScores.append((k, inertia_score))


feature_df = pd.DataFrame(testScores, columns=['K','Inertia Score'])
feature_df.plot(title='Relation Between K Value and Inertia Score', xlabel='K', ylabel='Inertia Score', xticks=range(0,10,1), legend=None)


# Elbow at k=3
kmeans = KMeans(n_clusters=3)
cluster_labels = kmeans.fit_predict(features)
df['cluster'] = cluster_labels

cluster_one = df[df.cluster == 0]
cluster_two = df[df.cluster == 1]
cluster_three = df[df.cluster == 2]

# Average scores in each cluster for each category
print('Cluster 1 Average Scores')
print('Math: ' + str(cluster_one['MathScore'].mean()))
print('Reading: ' + str(cluster_one['ReadingScore'].mean()))
print('Writing: ' + str(cluster_one['WritingScore'].mean()))

print('Cluster 2 Average Scores')
print('Math: ' + str(cluster_two['MathScore'].mean()))
print('Reading: ' + str(cluster_two['ReadingScore'].mean()))
print('Writing: ' + str(cluster_two['WritingScore'].mean()))

print('Cluster 3 Average Scores')
print('Math: ' + str(cluster_three['MathScore'].mean()))
print('Reading: ' + str(cluster_three['ReadingScore'].mean()))
print('Writing: ' + str(cluster_three['WritingScore'].mean()))


# Percentage of test prep
print('Cluster One')
print('-----------')
print((cluster_one['TestPrep'].value_counts(normalize=True)*100))
print('---------------------------------------------------------------------------')
print('Cluster One')
print(cluster_one.head(2))

print('Cluster Two')
print('-----------')
print((cluster_two['TestPrep'].value_counts(normalize=True)*100))
cluster_two.head()
print('---------------------------------------------------------------------------')
print('Cluster Two')
print(cluster_two.head(2))

print('Cluster Three')
print('-----------')
print((cluster_three['TestPrep'].value_counts(normalize=True)*100))
print('---------------------------------------------------------------------------')
print('Cluster Three')
print(cluster_three.head(2))
