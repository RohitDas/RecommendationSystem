from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.fpm import FPGrowth

conf = SparkConf()
context = SparkContext(conf=conf)
sql_context = SQLContext(context)


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




def similar_items_for_type(rdd, index, type):
    print("Calculating similar items for type: ", type)
    new_rdd = rdd.map(lambda row: [row[0]]+(row[index] if row[index] else []))
    new_rdd = new_rdd.map(lambda row: list(set(row)))
    model = FPGrowth.train(new_rdd, minSupport=0.01, numPartitions=4)
    freq_items_sets = model.freqItemsets().collect()
    item_to_sim = {}
    for freq_item_set in freq_items_sets:                                                 
        items = set(freq_item_set[0]) 
        for item in items:
            item_to_sim.setdefault(item, set()).update(items.difference(set([item])))
    return item_to_sim

print(similar_items_for_type(rdd_meta, 1, "also_viewed"))

#print(rdd_meta.collect())

#print("Count: ", count)


"""
    I need to find top k recommendation for each user.
"""


def average_also_width(rdd, ind, type):
    pass



