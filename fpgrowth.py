from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.fpm import FPGrowth

conf = SparkConf()
context = SparkContext(conf=conf)
sql_context = SQLContext(context)

/#The metadata section file
meta_f_name = "meta_Musical_Instruments.json"

#Load as a dataframe
meta = sql_context.read.json(meta_f_name)

#Get the count of unique ids
count = meta.count()

#Some asins are null removing them
filtered_meta = meta.filter('asin is not null')
rdd_meta = filtered_meta.select('asin', 'related').rdd
rdd_meta = rdd_meta.filter(lambda row: row['related']!=None)
rdd_meta = rdd_meta.map(lambda row: (row['asin'], row['related']))
rdd_meta = rdd_meta.map(lambda row: (row[0], row[1]['also_bought'], row[1]['also_viewed'], row[1]['bought_together'], row[1]['buy_after_viewing']))


#finding frequent itemsets

def similar_items_for_type(rdd, index, type):
    print("Calculating similar items for type: ", type)
    new_rdd = rdd.map(lambda row: [row[0]]+(row[index] if row[index] else []))
    new_rdd = new_rdd.map(lambda row: list(set(row)))
    model = FPGrowth.train(new_rdd, minSupport=0.001, numPartitions=4)
    freq_items_sets = model.freqItemsets().collect()
    item_to_sim = {}
    for freq_item_set in freq_items_sets:                                                 
        items = set(freq_item_set[0]) 
        for item in items:
            item_to_sim.setdefault(item, set()).update(items.difference(set([item])))
    return item_to_sim

item_to_sim = similar_items_for_type(rdd_meta, 2, "also_viewed")
print item_to_sim

#taking the review section of the data for partitioning into training and test sets respectively.

dfmain = sql_context.read.json ("reviews_Musical_Instruments.json")        
dfmain.show()                                                                                   
dfmain.createOrReplaceTempView("json_view")                                                     
dfnew=sql_context.sql("select reviewerID,asin,overall from json_view")                          
dfnew.show()                                                                                    
rdd_allreviews = dfnew.select('reviewerID', 'asin', 'overall').rdd                               
print rdd_allreviews.take(10)                                                                   
training_RDD,test_RDD = rdd_allreviews.randomSplit([5, 5], seed=0L)

#looking at the items a particular user has viewed
training_RDD_user_to_asins = training_RDD.map(lambda row: (row['reviewerID'], [row['asin']])).reduceByKey(lambda x,y: x + y).collect()
test_RDD_user_to_asins = test_RDD.map(lambda row: (row['reviewerID'], [row['asin']])).reduceByKey(lambda x,y: x + y).collect()

#Generation of recommendations

recommendations = {}


def get_reco_for_user(asins):
    all_asins = set()
    for asin in asins:
        all_asins.update(item_to_sim.get(asin, []))
    return all_asins


for user, asins in training_RDD_user_to_asins:
    recommendations.update({
        user: get_reco_for_user(asins)
    })

#Calculation of conversion rate

conversion_rate = 0
for user, asins in test_RDD_user_to_asins:
    reco_products = recommendations.get(user, set())
    if reco_products.intersection(set(asins)):
        conversion_rate += 1
        print "Reco successful"

print("Average conversion rate: ", float(conversion_rate)/float(len(test_RDD_user_to_asins)*100))


