# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import warnings
warnings.filterwarnings('ignore')
# Initialize Spark session
spark = SparkSession.builder.appName("LoanPrediction").getOrCreate()

# Load and preprocess data
data = spark.read.csv("Loan Prediction Dataset.csv", header=True, inferSchema=True)
data = data.dropna()

# Feature engineering
indexer = StringIndexer(inputCol="Loan_Status", outputCol="label")
data = indexer.fit(data).transform(data)

# Index and encode the Property_Area column
indexer = StringIndexer(inputCol="Property_Area", outputCol="Property_Area_Index")
data = indexer.fit(data).transform(data)

encoder = OneHotEncoder(inputCol="Property_Area_Index", outputCol="Property_Area_OHE")
data = encoder.fit(data).transform(data)

# Use relevant features
assembler = VectorAssembler(inputCols=["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area_OHE"], outputCol="features")
data = assembler.transform(data)

# Split data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Model selection and training
rf = RandomForestClassifier(featuresCol="features", labelCol="label")

# Hyperparameter tuning
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)

cvModel = crossval.fit(train_data)

# Model evaluation
predictions = cvModel.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Test Accuracy: {accuracy:.2f}")
