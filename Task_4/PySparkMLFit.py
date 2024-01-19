import operator
import argparse

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

MODEL_PATH = 'spark_ml_model'
LABEL_COL = 'is_bot'


def process(spark, data_path, model_path):
    """
    Основной процесс задачи.

    :param spark: SparkSession
    :param data_path: путь до датасета
    :param model_path: путь сохранения обученной модели
    """
    data = spark.read.parquet(data_path)
    
    indexer_user_type = StringIndexer(inputCol="user_type", outputCol="user_type_index")
    indexer_platform = StringIndexer(inputCol="platform", outputCol="platform_index")
    
    feature = VectorAssembler(
        inputCols=["user_type_index","duration","platform_index","item_info_events","select_item_events","make_order_events","events_per_min"],
        outputCol="features"
        )
    
    rf_classifier = RandomForestClassifier(labelCol=LABEL_COL, featuresCol="features")

    evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="f1")
    
    pipeline_rfc = Pipeline(stages=[indexer_user_type, indexer_platform, feature, rf_classifier])
    
    paramGrid_rfc = ParamGridBuilder()\
                  .addGrid(rf_classifier.maxDepth, [3, 4, 5])\
                  .addGrid(rf_classifier.maxBins, [5, 6, 7, 8])\
                  .addGrid(rf_classifier.minInfoGain, [0.001, 0.0015, 0.002, 0.01, 0.035, 0.05])\
                  .build()
                  
    tvs_rfc = TrainValidationSplit(estimator=pipeline_rfc,
                                   estimatorParamMaps=paramGrid_rfc,
                                   evaluator=evaluator,
                                   trainRatio=0.8)

    model_rfc = tvs_rfc.fit(data)
    model_rfc.bestModel.write().overwrite().save(model_path)


def main(data_path, model_path):
    spark = _spark_session()
    process(spark, data_path, model_path)


def _spark_session():
    """
    Создание SparkSession.

    :return: SparkSession 
    """
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='session-stat.parquet', help='Please set datasets path.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Please set model path.')
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    main(data_path, model_path)
