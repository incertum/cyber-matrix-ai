# `BigDL Deep Learning on Spark` 
# Detection of Malicious URLs

Melissa K (Dec 2017)

----------

Code
----------

Go right away to the [scala project src code folder](URL-Classification/src/main/scala/incertum/cybermatrixai/url/).

Requirements (see project's [build.sbt](URL-Classification/build.sbt)):

- `Spark 2.2.0` https://spark.apache.org/downloads.html
- `Scala 2.11.8` https://www.scala-lang.org/
- `BigDL 0.3.0` https://github.com/intel-analytics/BigDL

----------


Motivation
----------

`BigDL - Deep Learning on Spark` was first released by Intel Corporation on Dec 30, 2016. The [BigDL GitHub repo](https://github.com/intel-analytics/BigDL) also contains a couple of presentations. Alternatively checkout this [blogpost](https://software.intel.com/en-us/articles/bigdl-distributed-deep-learning-on-apache-spark) or the offical [BigDL Documentation](https://bigdl-project.github.io/master/) for more information. Besides source code documentation and the API Guide, there are currently very few open-sourced real-world projects. 

----------

Purpose and Limitations of the Project
----------

  - Provide a complete use case (transferrable to many Natural Language Processing related tasks).
  - Focus on pure `BigDL` implementation, no modeling justifications or implications will be discussed.
  - Share debugging and exploration experience.
  - Contrast to `Keras` implementation `See Python Keras URL Deep Learning project of this repo`. Got very similar model performance results when comparing BigDL to Keras.
  - Will update the project if I find more elegant ways. 

Absolute Deep Learning Beginners should first explore [Keras - Python](https://keras.io/) as there are so many great tutorials out there. 

----------


Preprocessing Raw URLs
----------

Below is a sample of the raw data loaded into a `Spark DataFrame`:

```scala
+--------------------+-----------+
|                 url|isMalicious|
+--------------------+-----------+
|kitchenaid.com/sh...|          0|
|ilike.com/artist/...|          0|
|about.com/style/t...|          0|
|             unc.edu|          0|
|ebay.com/1/711-53...|          0|
|telekom.com/en/in...|          0|
|aoyou.com/singlep...|          0|
|wiener-staatsoper...|          0|
|deproducts/fritza...|          0|
|flickr.com/2862/3...|          0|
|bpb.de/veranstalt...|          0|
|comatlassian-gadg...|          0|
|kcparksgolf.com/c...|          0|
|hhs.se/en/Researc...|          0|
|1soccer.com/views...|          0|
|cirabaonline.com/...|          1|
|starfruit.gr/osob...|          1|
|richardsonelectri...|          1|
|ataxiinlorca.com/...|          1|
|retipanzio.hu/kep...|          1|
+--------------------+-----------+
only showing top 20 rows
```

All preprocessing is done in [Driver.scala](URL-Classification/src/main/scala/incertum/cybermatrixai/url/Driver.scala) (using some methods from [Util.scala](URL-Classification/src/main/scala/incertum/cybermatrixai/url/Util.scala)). The main goal of the preprocessing stage is to tokenize the URLs characterwise while mapping each character to an index.

- Create a `HashMap` that contains relevant string characters as keys and unique indices as values.
- Convert this HashMap to a Spark `broadcast` variable.
- Resolve the broadcast variable to map each character of the URL to it's unique float type index starting from `3.0f`. Assign `2.0f` if character not available in Map. URLs that are longer than 75 characters will be cropped (`val sequenceLen = 75` was arbitrarily chosen by the researcher).
- Even though the Deep Learning embedding layer `LookupTable` provides a `padding` option for sequences that are shorter than the `sequenceLen`, I would get errors when supplying tensors of unequal lengths. Therefore, a spark `udf` (user defined function) that pads all shorter sequences with the index `1.0f` was added to the data preprocessing stage. 
- Thus vocabulary size in this project is `val vocabSize = 87`.



```scala
+--------------------+-----------+---------+--------------------+--------------------+----------------+
|                 url|isMalicious|urlLength|           urlTokens|          urlIndices|urlIndicesLength|
+--------------------+-----------+---------+--------------------+--------------------+----------------+
| liveinternet.ru/top|          0|       19|[l, i, v, e, i, n...|[24.0, 21.0, 34.0...|              75|
|kitchenaid.com/sh...|          0|       52|[k, i, t, c, h, e...|[23.0, 21.0, 32.0...|              75|
|democratandchroni...|          0|       39|[d, e, m, o, c, r...|[16.0, 17.0, 25.0...|              75|
|aviva.com/about-u...|          0|       79|[a, v, i, v, a, ....|[13.0, 34.0, 21.0...|              75|
|cuatro.com/notici...|          0|       77|[c, u, a, t, r, o...|[15.0, 33.0, 13.0...|              75|
|          fox4kc.com|          0|       10|[f, o, x, 4, k, c...|[18.0, 27.0, 36.0...|              75|
|eurosport.com/for...|          0|       59|[e, u, r, o, s, p...|[17.0, 33.0, 30.0...|              75|
|sacurrent.com/san...|          0|       51|[s, a, c, u, r, r...|[31.0, 13.0, 15.0...|              75|
|gulfnews.com/in-f...|          0|       52|[g, u, l, f, n, e...|[19.0, 33.0, 24.0...|              75|
|ballparks.com/NBA...|          0|       46|[b, a, l, l, p, a...|[14.0, 13.0, 24.0...|              75|
|thesitewizard.com...|          0|       73|[t, h, e, s, i, t...|[32.0, 20.0, 17.0...|              75|
|boston.com/boston...|          0|      102|[b, o, s, t, o, n...|[14.0, 27.0, 31.0...|              75|
|acus.org/programs...|          0|       63|[a, c, u, s, ., o...|[13.0, 15.0, 33.0...|              75|
|treas.gov/careers...|          0|       48|[t, r, e, a, s, ....|[32.0, 30.0, 17.0...|              75|
|flyanglersonline....|          0|       47|[f, l, y, a, n, g...|[18.0, 24.0, 37.0...|              75|
|anokaramsey.edu/n...|          0|       89|[a, n, o, k, a, r...|[13.0, 26.0, 27.0...|              75|
|ilike.com/artist/...|          0|       65|[i, l, i, k, e, ....|[21.0, 24.0, 21.0...|              75|
|jamieoliver.com/r...|          0|       52|[j, a, m, i, e, o...|[22.0, 13.0, 25.0...|              75|
|southbankcentre.c...|          0|       60|[s, o, u, t, h, b...|[31.0, 27.0, 33.0...|              75|
|popularresistance...|          0|       31|[p, o, p, u, l, a...|[28.0, 27.0, 28.0...|              75|
+--------------------+-----------+---------+--------------------+--------------------+----------------+
only showing top 20 rows
```
----------


Deep Learning Architectures in BigDL
----------

Before explaining next steps in data preparation and actual training, let's look at the neural network architectures. Similar to the `Python Keras URL Deep Learning Project of this repo` three example architectures are provided. 

Running `def debugArchitectures()` (see [DeepLearning.scala](URL-Classification/src/main/scala/incertum/cybermatrixai/url/DeepLearning.scala) class) will debug the neural network architectures via simulating a (`batch size` x `time`) input tensor. Highly recommend this simulation to ensure tensor dimensions match up at each layer of the network.

```scala
val rnd = new scala.util.Random(seed = 540)
// Create Dummy Input via simulating input  tensor (batch size x time)
// Input indices should not be equal to 0.0f, therefore adding 1.0f as random int include 0.0f
val inputDebug = Tensor(batchSizeValue, sequenceLen).apply1(e => rnd.nextInt(86) + 1.0f)
```

The [BigDL Layers Documentation](https://bigdl-project.github.io/master/#APIGuide/Layers/Embedding-Layers/) is complete, however contains only simple examples. Definitely recommending skimming through all of them! Below are code snippets for each of the example architectures ...

Having the `Embedding Layer` [BigDL LookupTable](https://bigdl-project.github.io/master/#APIGuide/Layers/Embedding-Layers/#lookuptable) as part of the architecture works better than training for example a [`word2vec`](https://spark.apache.org/docs/2.2.0/ml-features.html#word2vec) separately!


1. **Simple LSTM**

```scala
def getLSTMModelArchitecture(vocabSize: Int = 87, embeddingDim: Int = 32, lstmOutputSize: Int = 32, classNum: Int = 1) =
  Sequential()
    .add(LookupTable(vocabSize, embeddingDim, 1.0f))
    .add(Recurrent().add(LSTM(embeddingDim, lstmOutputSize)))
    .add(TimeDistributed(Linear(lstmOutputSize, classNum)))
    .add(Dropout(0.5))
    .add(Select(2, -1)) // select last LSTM output from time axis
    .add(Sigmoid())

Sequential[533a8815]{
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
  (1): LookupTable[c51c6577](nIndex=87,nOutput=32,paddingValue=1.0,normType=2.0)
  (2): Recurrent[134e80b]ArrayBuffer(TimeDistributed[d19ba6a3]Linear[b4476f01](32 -> 128), LSTM(32, 32, 0.0))
  (3): TimeDistributed[fb696a73]Linear[5557ac09](32 -> 1)
  (4): Dropout[911ce8da](0.5)
  (5): nn.Select
  (6): Sigmoid[8cdc11bb]
}

```


2. **1D Convolution and LSTM**

```scala
def get1DConvLSTMModelArchitecture(vocabSize: Int = 87, embeddingDim: Int = 32,
                                     outputFrameSize: Int = 256,
                                     lstmOutputSize: Int = 32, kernelW: Int = 5, poolSize: Int = 4,
                                     classNum: Int = 1) =
  Sequential()
    .add(LookupTable(vocabSize, embeddingDim, 1.0f))
    .add(TemporalConvolution(embeddingDim, outputFrameSize, kernelW, 1))
    .add(ELU())
    .add(Dropout(0.5))
    .add(Recurrent().add(LSTM(outputFrameSize, lstmOutputSize)))
    .add(TimeDistributed(Linear(lstmOutputSize, classNum)))
    .add(Dropout(0.5))
    .add(Select(2, -1)) // select last LSTM output from time axis
    .add(Sigmoid())

Sequential[1501386c]{
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> output]
  (1): LookupTable[7ced8090](nIndex=87,nOutput=32,paddingValue=1.0,normType=2.0)
  (2): nn.TemporalConvolution(32 -> 256, 5 x 1)
  (3): ELU[5eb3a722]
  (4): Dropout[80bbfcc9](0.5)
  (5): Recurrent[8d3e9c9]ArrayBuffer(TimeDistributed[4edfaca2]Linear[3f9dacd3](256 -> 128), LSTM(256, 32, 0.0))
  (6): TimeDistributed[5489e7ca]Linear[206bb8a1](32 -> 1)
  (7): Dropout[7b42e3](0.5)
  (8): nn.Select
  (9): Sigmoid[8a3eaa5c]
}

```



3. **1D Convolutions and Fully Connected Layers**

Using a very similar architecture that was proposed by Josh Saxe (see [blogpost](https://www.invincea.com/2017/02/look-ma-no-features-deep-learning-methods-in-intrusion-detection/) or full [paper](https://arxiv.org/pdf/1702.08568.pdf?lipi=urn%3Ali%3Apage%3Ad_flagship3_detail_base%3Bw%2BJ3QVESSsmjKlHtncWivw%3D%3D)).

```scala
def get1DConv(inputFrameSize: Int = 32, outputFrameSize: Int = 256, kernelW: Int = 5) =
  Sequential()
    .add(TemporalConvolution(inputFrameSize, outputFrameSize, kernelW, 1))
    .add(ELU())
    .add(Sum(dimension = 2))
    .add(Dropout(0.5))

def getConcatConv() =
  Concat(dimension = 2)
    .add(get1DConv(kernelW = 2))
    .add(get1DConv(kernelW = 3))
    .add(get1DConv(kernelW = 4))
    .add(get1DConv(kernelW = 5))

def get1DConvFullyModelArchitecture(vocabSize: Int = 87, embeddingDim: Int = 32, classNum: Int = 1) =
  Sequential()
    .add(LookupTable(vocabSize, embeddingDim))
    .add(getConcatConv())
    .add(Linear(1024, 1024))
    .add(ELU())
    .add(Linear(1024, classNum))
    .add(Sigmoid())

Sequential[e1968b73]{
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
  (1): LookupTable[87a8d091](nIndex=87,nOutput=32,paddingValue=0.0,normType=2.0)
  (2): Concat[a39904c5]{
    input
      |`-> (1): Sequential[9f7f42fe]{
      |      [input -> (1) -> (2) -> (3) -> (4) -> output]
      |      (1): nn.TemporalConvolution(32 -> 256, 2 x 1)
      |      (2): ELU[f01820d9]
      |      (3): nn.Sum
      |      (4): Dropout[4cf1eb98](0.5)
      |    }
      |`-> (2): Sequential[3cb24d85]{
      |      [input -> (1) -> (2) -> (3) -> (4) -> output]
      |      (1): nn.TemporalConvolution(32 -> 256, 3 x 1)
      |      (2): ELU[4e382bb]
      |      (3): nn.Sum
      |      (4): Dropout[980e3489](0.5)
      |    }
      |`-> (3): Sequential[93ff6c57]{
      |      [input -> (1) -> (2) -> (3) -> (4) -> output]
      |      (1): nn.TemporalConvolution(32 -> 256, 4 x 1)
      |      (2): ELU[13d2224a]
      |      (3): nn.Sum
      |      (4): Dropout[ef2e1d11](0.5)
      |    }
      |`-> (4): Sequential[9f690393]{
             [input -> (1) -> (2) -> (3) -> (4) -> output]
             (1): nn.TemporalConvolution(32 -> 256, 5 x 1)
             (2): ELU[2bcaf89f]
             (3): nn.Sum
             (4): Dropout[5d6790bb](0.5)
           }
       ... -> output
    }
  (3): Linear[bcbb5f21](1024 -> 1024)
  (4): ELU[c2aaa7a4]
  (5): Linear[b753bbc9](1024 -> 1)
  (6): Sigmoid[7c8e5597]
}

```
----------

BigDL Input Data Format
----------

There are various [BigDL Data Types](https://bigdl-project.github.io/master/#APIGuide/Data/). When using the **`RDD based API`** a `RDD` of Type `Sample` with `Tensors` for both `feature` and `label` has to be created first, similar to below. For demonstration purposes the DataFrame is first mapped to a DataSet, then to a RDD. Loading the data and preprocessing could have also been done using RDD manipulations.

```scala
// Create BigDL Samples Tensors from DataFrame
val df1 = df.select($"urlIndices", $"isMalicious".cast("float"))

val rddSamples = df1.as[(Array[Float], Float)].rdd
  .map { case (indices: Array[Float], target: Float) =>
    Sample(
      featureTensor = Tensor(Storage(indices), 1, Array(indices.size)),
      labelTensor = Tensor(Array(target), Array(classNum))
    )
  }
```

When using the **`DataFrame based API`** ensure that neither `label` or `feature` column contain any `null` values and that the `label` column is of type `Array[Float]`.

```scala
val df2 = df
  .withColumn("isMalicious", $"isMalicious".cast("float")).na.fill(value = 0.0f, cols = Seq("isMalicious"))
  .select($"urlIndices", array($"isMalicious").alias("isMalicious"))
  
// Schema  
root
|-- urlIndices: array (nullable = true)
|    |-- element: float (containsNull = false)
|-- isMalicious: array (nullable = false)
|    |-- element: float (containsNull = false)
```





----------


BigDL Training 
----------

Running `def trainModels()` (see [DeepLearning.scala](URL-Classification/src/main/scala/incertum/cybermatrixai/url/DeepLearning.scala) class) will start the training. Change model architecture or parameters such as `maxEpochValue` etc for model performance assessment.

**Training using RDD API**

The RDD based API appears to be faster and more flexible in terms of methods available, so if you are using DataFrames in your project (which I usually prefer), simply convert them to RDDs as shown above in the `BigDL Input Data Format` section. 

Simple cross-validation:

```scala
// Simple Cross-Validation: Train - Validation Split
val Array(rddSamplesTrain, rddSamplesVal) = rddSamples.randomSplit(
  Array(0.8, 1 - 0.8))
```

Initialize the `optimizer`. For example supply the `Sequential Deep Learning model` and since here it is a binary classfication task set the `loss` to binary cross-entropy `BCECriterion`. Next set the optimization method (here for example `Adam`). Finally calling `optmize()` will start the training. For evaluation `Top1Accuracy` is just the regular accuracy score. Now new predictions can be made. See [BigDL Model Guide](https://bigdl-project.github.io/master/#APIGuide/Module/) for additional options such as saving and loading a model.

```scala
val vocabSize = 87
val sequenceLen = 75
val embeddingDim = 32
val batchSizeValue = 36 // BigDL requires the batch size to be a multiple of nodeNumber * coreNumber
val maxEpochValue = 1
val classNum = 1 // Binary Classification Task
    
// Get DL model
val modelDL = get1DConvFullyModelArchitecture()

val optimizer = Optimizer(
  model = modelDL,
  sampleRDD = rddSamplesTrain,
  criterion = new BCECriterion[Float](),
  batchSize = batchSizeValue
)

optimizer
  .setOptimMethod(new Adam(learningRate = 1e-4, learningRateDecay = 0.0, beta1 = 0.9, beta2 = 0.999, Epsilon = 1e-8))
  .setValidation(Trigger.everyEpoch, rddSamplesVal, Array(new Top1Accuracy[Float]().asInstanceOf[ValidationMethod[Float]]), batchSizeValue)
  .setEndWhen(Trigger.maxEpoch(maxEpochValue))
  .optimize()

val evaluateResult = modelDL.evaluate(rddSamplesVal, Array(new Top1Accuracy[Float]().asInstanceOf[ValidationMethod[Float]]), None)
println(s"BigDL Test Complete Architecture Evaluation: ${evaluateResult.mkString(",")}")

println("Making new predictions")
val newPredictions = modelDL.predict(rddSamplesVal)

```

**Training using DataFrame API**

```scala
val criterion2 = new BCECriterion[Float]()
val optim = new Adam(learningRate = 1e-4, learningRateDecay = 0.0, beta1 = 0.9, beta2 = 0.999, Epsilon = 1e-8)

val dlClf = new DLClassifier(modelDL, criterion2, Array(sequenceLen))
  .setFeaturesCol("urlIndices")
  .setLabelCol("isMalicious")
  .setBatchSize(batchSizeValue)
  .setOptimMethod(optim)
  .setMaxEpoch(maxEpochValue)

// Fit DL model
val dlModel = dlClf.fit(df2)
// Predict
dlModel.transform(df2).select($"isMalicious", $"prediction").show(false)
```
----------



Spark Info and Building Project using SBT
----------

SparkInit trait (see [SparkInit.scala](URL-Classification/src/main/scala/incertum/cybermatrixai/url/SparkInit.scala) class)  creates a `SparkContext`, `SQLContext` and `SparkSession` over the `BigDL Engine`. It's currently set up for local development. Most important is that the Deep Learning `batch size` has to be a multiple of nodeNumber * coreNumber!   

See [BigDL Install Guide](https://bigdl-project.github.io/master/#ScalaUserGuide/install-pre-built/) or the projects [build.sbt](URL-Classification/build.sbt) file. Project's `Main` class is `Driver` (see [Driver.scala](URL-Classification/src/main/scala/incertum/cybermatrixai/url/Driver.scala)) 

**Build using SBT**

- Navigate to project root folder `URL-Classification`
- Make sure [`SBT`](https://www.scala-sbt.org/) is installed, now run `sbt assembly` from the command line.
- Path of `.jar` is `URL-Classification/target/scala-2.11/URL-Classification-assembly-0.1.jar`
- From the project root run `spark-submit --class incertum.cybermatrixai.url.Driver target/scala-2.11/URL-Classification-assembly-0.1.jar` for a **test run on localhost**.
- Note that Spark has to be installed on your system and added to your path (e.g `export PATH=$SPARK_HOME/bin:$PATH` because of the `% "provided"` in [build.sbt](URL-Classification/build.sbt).
- When for example running the Application in `IntelliJ IDE` use the version without "provided".

```scala
libraryDependencies ++= {
  Seq(
    "com.typesafe.play" %% "play-json" % "2.3.8",
    "org.apache.spark" %% "spark-core" % sparkVer % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVer % "provided",
    "org.apache.spark" %% "spark-sql" % sparkVer % "provided",
    "org.apache.spark" %% "spark-streaming" % sparkVer % "provided"
  )
}
```


----------



Other Distributed Deep Learning Options for Spark
----------


- CaffeOnSpark https://github.com/yahoo/CaffeOnSpark

- DeepDist https://github.com/dirkneumann/deepdist/

- Deeplearning4J https://deeplearning4j.org/spark

- TensorFlowOnSpark https://github.com/yahoo/TensorFlowOnSpark

- TensorFrames https://github.com/databricks/tensorframes

----------
