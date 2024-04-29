package org.example.winepredictionapp;

import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class App {
    public static final Logger logger = LoggerFactory.getLogger(App.class);

    private static final String ACCESS_KEY_ID = "ASIA5FTY7O2RKB24LIKH";
    private static final String SECRET_KEY = "P+VLqH9hM2vWTQvOEEzr9xFl6CLVSGFtL8frDY8l";

    private static final String MASTER_URI = "local[*]";

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction App").master(MASTER_URI)
                .config("spark.executor.memory", "3g")
                .config("spark.driver.memory", "3g")
                .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.2")
                .getOrCreate();

        spark.sparkContext().hadoopConfiguration().set("fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain");
        spark.sparkContext().hadoopConfiguration().set("fs.s3a.access.key", ACCESS_KEY_ID);
        spark.sparkContext().hadoopConfiguration().set("fs.s3a.secret.key", SECRET_KEY);

        LogisticRegressionV2 parser = new LogisticRegressionV2();
        parser.predict(spark);

        spark.stop();
    }
}
