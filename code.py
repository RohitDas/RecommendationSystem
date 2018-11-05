from pyspark import SparkConf, SparkContext, SQLContext

conf = SparkConf()

context = SparkContext(conf=conf)

sql_context = SQLContext(context)

dfmain = sql_context.read.json ("/home/spark/Desktop/reviews_Musical_Instruments.json")		//load data into a dataframe

dfmain.show()											//validation to see whether data has loaded properly

dfmain.createOrReplaceTempView("json_view")							//creating a temp view from the dataframe

dfnew=sql_context.sql("select reviewerID,asin,overall from json_view")				//forming a new dataframe with the desired fields

dfnew.show()											//validation to see whether data has loaded properly

rdd_allreviews = dfnew.select('reviewerID', 'asin', 'overall').rdd				//converting the dataframe into a rdd

print rdd_allreviews.take(10)									//printing the first 10 rows to verify whether data has been loaded properly

training_RDD,test_RDD = rdd_allreviews.randomSplit([8, 2], seed=0L)				//splitting the original rdd into train and test using the randomsplit function.	
