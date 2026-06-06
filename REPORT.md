# Streaming ML: Credit Card Fraud Detection with Spark ML

**Chris Thompson**
Parallel and Distributed Computing, Assignment 2

## Overview of the Data

The dataset for this project is the Credit Card Fraud Detection dataset published by the Machine Learning Group at the Université Libre de Bruxelles, available on Kaggle. It contains transactions made by European cardholders over two days in September 2013. In total there are 284,807 transactions, of which only 492 are fraudulent. This makes the data extremely imbalanced: the fraud class accounts for roughly 0.17 percent of all records, or about one fraudulent transaction for every 578 legitimate ones.

The features are almost entirely numeric. Columns V1 through V28 are principal components produced by a PCA transformation of the original transaction attributes, which the publishers anonymized for confidentiality. The only two columns left in their original form are `Time`, the number of seconds elapsed between each transaction and the first transaction in the set, and `Amount`, the transaction value. The target column is `Class`, which is 1 for fraud and 0 for a legitimate transaction.

This dataset suits the assignment well for two reasons. First, the PCA features are already clean and numeric, so the work of the project lives in the Spark pipeline and the streaming setup rather than in heavy data cleaning. Second, the `Time` column gives a natural ordering that lets the streaming portion be more than a simulation gimmick. Fraud detection in the real world is exactly the problem of building a model on past transactions and then scoring new transactions as they arrive, so streaming the later records past a model trained on the earlier ones reflects the authentic use case.

## The Machine Learning Problem

The goal is binary classification: given a transaction, predict whether it is fraudulent. The intellectual center of this problem is the class imbalance, and nearly every design decision in the project follows from it.

### Exploratory analysis

Before modeling, I characterized the data along the dimensions that would shape the pipeline. The imbalance was quantified directly: 492 fraud records out of 284,807, which is 0.1727 percent of the data. This single number justifies two later choices, the move away from accuracy as a metric and the use of class weighting during training.

A correlation screen between each feature and the `Class` label showed that several PCA components carry a meaningful linear signal. The strongest negative relationships were with V17, V14, V12, and V10, and the strongest positive relationships were with V11 and V4. These were treated as a feature screening exercise rather than a basis for dropping anything. Correlation captures only the linear or monotonic part of a relationship, and because the V features are PCA components they are largely orthogonal to one another, so a low correlation with the label does not mean a feature is useless. All features were kept so the model could exploit nonlinear structure.

I also compared the `Amount` distribution across the two classes. Both classes are heavily right skewed, with a mean well above the median, which indicates that most transactions are small while a minority of large transactions pull the average up. Fraudulent transactions had a lower median amount (9.21 versus 22.00) but a higher mean (122.21 versus 88.29), and a much lower maximum (2,125.87 versus 25,691.16). The interpretation is that fraud tends to cluster in the small to moderate range, consistent with card testing behavior, while legitimate activity occasionally produces very large transactions that fraud does not. The wide scale difference between `Amount` and the V features is also the direct motivation for the standardization step in the pipeline.

### The pipeline

I built two Spark ML pipelines with an identical preprocessing path and a different final estimator, so the two models could be compared on equal footing. The feature columns are every raw column except `Time` and `Class`. `Time` is deliberately excluded as a model input because it is the key used to define the train and test split; using it as a predictor would entangle the feature space with the split itself. The pipeline stages are a `VectorAssembler` to collect the feature columns into a single vector, a `StandardScaler` to put the features on a comparable scale, and then the classifier.

The first pipeline ends in a `LogisticRegression` estimator and the second in a `SparkXGBClassifier`. Both were fit on the training data to produce fitted pipeline models ready to transform new data.

### Handling the imbalance with class weights

The one deliberate refinement in this project is class weighting, kept inside the Spark paradigm rather than handled by resampling outside it. I added a `classWeight` column using inverse frequency, so that the rare fraud rows receive a large weight (about 0.998) and the common legitimate rows receive a tiny weight (about 0.0018). Logistic regression learns by minimizing a loss that sums each row's error. With the weight applied, each row's error term is scaled by its weight, so a fraud row contributes roughly 578 times as much to the loss as a legitimate row. This makes the optimizer far more sensitive to getting fraud cases right, which is the correct tradeoff for this problem: a false negative (a missed fraud) is more costly than a false positive (a legitimate transaction flagged for review). The same idea was passed to the XGBoost estimator through its `weight_col` parameter. The weight affects training only and plays no role at prediction time, so the streamed data never needs the weight column.

### Results

Because of the imbalance, accuracy is not a meaningful metric. A model that simply predicts "legitimate" for every transaction would score above 99.8 percent accuracy while catching no fraud at all. I therefore led with the area under the precision recall curve (PR AUC), along with the precision and recall on the fraud class.

On the held out test set, evaluated as a batch before streaming, the logistic regression model reached a PR AUC of 0.7301 and an ROC AUC of 0.9863. Its confusion matrix showed 68 frauds caught and 7 missed, at the cost of 1,478 legitimate transactions flagged. The XGBoost model reached a higher PR AUC of 0.7731 with an ROC AUC of 0.9821. Its confusion matrix showed 60 frauds caught and 15 missed, with only 140 legitimate transactions flagged.

The two models illustrate a genuine tradeoff. Logistic regression caught more frauds (higher recall) but raised many more false alarms, while XGBoost caught slightly fewer frauds but with far better precision. Since XGBoost was the stronger model on the headline PR AUC metric, by about four percentage points, it was the model saved and used for the streaming stage.

## The Streaming Section

The streaming design follows the assignment's suggested approach while keeping the train and test boundary honest. Rather than a random split, I split the data on `Time` at the 80th percentile: the earlier 80 percent of transactions became the historical training data, and the later 20 percent were held out to be streamed. This avoids temporal leakage and mirrors how a deployed fraud model actually operates, learning from the past and scoring the future.

The fitted XGBoost pipeline was saved to disk with Spark ML's native save method. To simulate a live feed, the held out test set was ordered by `Time` and written to ten separate CSV files, each a contiguous time slice. Splitting the small test set into files on the driver is simulation plumbing rather than analysis, so it is the one place a brief use of pandas is appropriate; the actual streaming source and model inference remain pure Spark.

The streaming job reloads the saved pipeline model and reads the directory of files as a Structured Streaming source. File based streaming requires an explicit schema, so the schema from the original DataFrame was reused. Setting `maxFilesPerTrigger` to 1 guarantees that exactly one file is consumed per trigger, so each of the ten files becomes one micro batch in time order. Inference and evaluation are handled inside a `foreachBatch` function. This pattern is important: inside `foreachBatch` each micro batch is an ordinary batch DataFrame, which lets the saved pipeline transform it cleanly and sidesteps the friction that can arise when calling the XGBoost Spark model's transform directly on a streaming DataFrame. For each batch the function computes the confusion counts and reports PR AUC, precision, and recall.

The per batch results confirmed the stream worked and showed the kind of variance expected when evaluating tiny slices of an imbalanced dataset. Recall stayed high across most batches, ranging from 0.40 to 1.00, while precision swung more widely because each batch contains only a handful of true frauds, so a few false positives move the precision sharply. Batches 7 and 8 in particular showed low precision and PR AUC, a reminder that metrics computed on very small windows are noisy and are best read alongside the overall batch evaluation rather than in isolation.

## Issues Encountered

Several practical issues came up across data preparation, modeling, and streaming.

The first was a library compatibility break. Importing `pyspark.pandas` failed because the installed PySpark version expected a private pandas internal that a newer pandas release had removed. The resolution was to avoid `pyspark.pandas` entirely and use either plain pandas for small exploratory work or native Spark APIs, which sidestepped the broken module.

The second was the imbalance itself, which forced a change in how success was measured. Early on it became clear that accuracy was misleading, and the project switched to PR AUC and fraud class recall and precision. This was not just a metric swap but a reframing of the whole evaluation around the rare positive class.

The third was specific to streaming. File based streaming sources do not infer their schema, so an explicit schema had to be supplied. Evaluating an imbalanced stream in small per file batches also produced noisy metrics, which had to be interpreted carefully rather than taken at face value. Finally, the XGBoost Spark model required the `foreachBatch` inference pattern to transform streaming data smoothly, and the missing parameter had to be set so the estimator would train correctly.

A final logistical issue arose outside the modeling work. The raw dataset is large enough that committing it to version control caused repeated failures when pushing to the remote repository, including transfer corruption on a pack of well over one hundred megabytes. The fix was to stop tracking the dataset in version control and reference its public Kaggle source instead, which kept the repository small and reproducible.

## Conclusion

This project built and compared two Spark ML pipelines for credit card fraud detection, addressed the severe class imbalance with inverse frequency class weighting, and demonstrated Spark Structured Streaming by training on historical transactions and scoring later transactions one file per trigger. The XGBoost pipeline was the stronger model with a test PR AUC of 0.7731, and it maintained high recall on fraud throughout the streamed evaluation, which is the behavior that matters most for catching fraud in practice.