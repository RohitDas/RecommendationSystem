from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

conf = SparkConf()
sc = SparkContext(conf=conf)
sql = SQLContext(sc)
reviews = sql.read.json("reviews_Musical_Instruments.json")
reviews = reviews.select("reviewerID", "asin", "overall") 

print("DISTINCT: ", reviews.select("reviewerID").distinct().count())
rdd = reviews.rdd                               

train, test = rdd.randomSplit(weights=[0.8,0.2], seed=42)

all_uids = rdd.map(lambda row: row['reviewerID']).collect()
all_asins = rdd.map(lambda row: row['asin']).collect()

uid_to_index, asin_to_index = {}, {}

for uid in all_uids:                                                                            
    if uid not in uid_to_index:
        uid_to_index[uid] = len(uid_to_index)

for asin in all_asins:                                                                          
    if asin not in asin_to_index:
        asin_to_index[asin] = len(asin_to_index)

train_rate = train.map(lambda row: Rating(uid_to_index[row[0]], asin_to_index[row[1]], row[2]))
test_rate = test.map(lambda row: Rating(uid_to_index[row[0]], asin_to_index[row[1]], row[2]))

#Rating rdd:
rating_rdd = rdd.map(lambda row: Rating(uid_to_index[row[0]], asin_to_index[row[1]], row[2]))

#Best model for param.
ranks = [16]
numIterations = 2
best_model, min_mse = None, None
for rank in ranks:
    print("Rank: ", rank)
    model = ALS.train(train_rate, rank, numIterations)
    testdat = test_rate.map(lambda x: (x[0], x[1]))
    predictions = model.predictAll(testdat).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = test_rate.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    if min_mse is None or MSE < min_mse:
        min_mse = MSE
        best_model = model


#Best model on the entire dataset
best_model_total, min_mse = None, None
for rank in ranks:
    print("Rank: ", rank)
    model = ALS.train(rating_rdd, rank, numIterations)
    testdat = rating_rdd.map(lambda x: (x[0], x[1]))
    predictions = model.predictAll(testdat).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = test_rate.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    if min_mse is None or MSE < min_mse:
        min_mse = MSE
    best_model_total = model


#Calculate the recommendations.
k = 10
#print("User recommendations: ", user_recommendations.collect())

"""
    Evaluation model.
    1. find user to production list in the test set.
    2. Use the recommendatiion for users to get the intersection set.
"""
def is_there_an_intersection(user_product_buys):
    pass


uid_to_asins_from_test = test.map(lambda row: (uid_to_index[row['reviewerID']], [asin_to_index[row['asin']]])).reduceByKey(lambda x,y: x+y).map(lambda row: (row[0], set(row[1]))).collect()
uid_to_asins_total = rdd.map(lambda row: (uid_to_index[row['reviewerID']], [asin_to_index[row['asin']]])).reduceByKey(lambda x,y: x+y).map(lambda row: (row[0], set(row[1]))).collect()


conversion_rate = 0
idx = 0
for uid, asins in uid_to_asins_from_test:
    if idx % 100 == 0:
        print "idx: ", idx, "conversion_rate: ", conversion_rate
    try:
        recommended_products = set([reco.product for reco  in best_model.recommendProducts(uid, k)])
        recommended_products_total = set([reco.product for reco in best_model_total.recommendProducts(uid, k)])
    except:
        recommended_products = set()
        try:
            recommended_products_total = set([reco.product for reco in best_model_total.recommendProducts(uid, k)])
        except:
            recommended_products_total = set()
    if asins.intersection(recommended_products_total):
        conversion_rate += 1

print("Average conversion rate: ", conversion_rate/len(uid_to_asins_from_test))






print("Best model: ", min_mse)





