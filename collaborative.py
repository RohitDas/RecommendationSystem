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

train = train.map(lambda row: Rating(uid_to_index[row[0]], asin_to_index[row[1]], row[2]))
test = test.map(lambda row: Rating(uid_to_index[row[0]], asin_to_index[row[1]], row[2]))

rank = 15                                                                                   
numIterations = 20

model = ALS.train(train, rank, numIterations)  

print("Model is trained")
#Error on TRAIN

traindat = train.map(lambda x: (x[0], x[1])) 
predictions = model.predictAll(traindat).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = train.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

testdat = test.map(lambda x: (x[0], x[1]))
predictions = model.predictAll(testdat).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = test.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))
