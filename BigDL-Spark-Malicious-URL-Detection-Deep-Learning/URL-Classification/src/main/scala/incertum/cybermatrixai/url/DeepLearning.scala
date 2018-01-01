package incertum.cybermatrixai.url

import Driver._

// BigDL Deep Learning on Spark
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

// Spark ML Library DataFrame API
import org.apache.spark.ml._
import org.apache.spark.ml.feature.Word2Vec

// Spark SQL
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import spark.implicits._

import scala.util.Random

/**
  * Contributors: incertum (December, 2017)
  * BigDL Documentation: https://bigdl-project.github.io/master/
  */

class DeepLearning {


  def getLSTMModelArchitecture(vocabSize: Int = 87, embeddingDim: Int = 32, lstmOutputSize: Int = 32, classNum: Int = 1) =
    Sequential()
      .add(LookupTable(vocabSize, embeddingDim, 1.0f))
      .add(Recurrent().add(LSTM(embeddingDim, lstmOutputSize)))
      .add(TimeDistributed(Linear(lstmOutputSize, classNum)))
      .add(Dropout(0.5))
      .add(Select(2, -1)) // select last LSTM output from time axis
      .add(Sigmoid())


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
      .add(LookupTable(vocabSize, embeddingDim, 1.0f))
      .add(getConcatConv())
      .add(Linear(1024, 1024))
      .add(ELU())
      .add(Linear(1024, classNum))
      .add(Sigmoid())

  def trainModels(df: DataFrame): Unit = {

    val sequenceLen = 75
    val embeddingDim = 32
    val batchSizeValue = 36 // BigDL requires the batch size to be a multiple of nodeNumber * coreNumber
    val maxEpochValue = 1
    val classNum = 1 // Binary Classification Task

    // Create BigDL Samples Tensors from DataFrame
    val df1 = df.select($"urlIndices", $"isMalicious".cast("float"))

    val rddSamples = df1.as[(Array[Float], Float)].rdd
      .map { case (indices: Array[Float], target: Float) =>
        Sample(
          featureTensor = Tensor(Storage(indices), 1, Array(indices.size)),
          labelTensor = Tensor(Array(target), Array(classNum))
        )
      }

    // Simple Cross-Validation: Train - Validation Split

    val Array(rddSamplesTrain, rddSamplesVal) = rddSamples.randomSplit(
      Array(0.8, 1 - 0.8))

    // Get DL model
    //    val modelDL = get1DConvFullyModelArchitecture()
    val modelDL = get1DConvLSTMModelArchitecture()
    //    val modelDL = getLSTMModelArchitecture()
    println(modelDL)

    // Using RDD API for training the model

    println("!!! Training Deep Learning Classifier using RDD API...")

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
    //    newPredictions.foreach(println)

    // Using DataFrame API for training the model

    //    println("!!! Training Deep Learning Classifier using DataFrame API...")
    //
    //    val df2 = df
    //      .withColumn("isMalicious", $"isMalicious".cast("float")).na.fill(value = 0.0f, cols = Seq("isMalicious"))
    //      .select($"urlIndices", array($"isMalicious").alias("isMalicious"))
    //    //    df2.show()
    //    //    df2.printSchema
    //
    //    val criterion2 = new BCECriterion[Float]()
    //    val optim = new Adam(learningRate = 1e-4, learningRateDecay = 0.0, beta1 = 0.9, beta2 = 0.999, Epsilon = 1e-8)
    //
    //    val dlClf = new DLClassifier(modelDL, criterion2, Array(sequenceLen))
    //      .setFeaturesCol("urlIndices")
    //      .setLabelCol("isMalicious")
    //      .setBatchSize(batchSizeValue)
    //      .setOptimMethod(optim)
    //      .setMaxEpoch(maxEpochValue)
    //
    //    // Fit DL model
    //    val dlModel = dlClf.fit(df2)
    //    // Predict
    //    dlModel.transform(df2).select($"isMalicious", $"prediction").show(false)

  }


  def debugArchitectures(): Unit = {

    // Input word2vec embedding parameters
    val batchSizeValue = 36
    val sequenceLen = 75
    val embeddingDim = 32
    val classNum = 1

    val rnd = new scala.util.Random(seed = 540)
    // Create Dummy Input via simulating input  tensor (batch size x time)
    // Input indices should not be equal to 0.0f, therefore adding 1.0f as random int include 0.0f
    val inputDebug = Tensor(batchSizeValue, sequenceLen).apply1(e => rnd.nextInt(86) + 1.0f)
    println("Debugging model architecture Input ...")
    println(inputDebug)

    // Debugging Simple LSTM
    val modelLSTM = getLSTMModelArchitecture()
    println(modelLSTM)

    // Debugging 1D Convolutions and Fully Connected Layers
    val model1DConvLSTM = get1DConvLSTMModelArchitecture()
    println(model1DConvLSTM)

    // Debugging 1D Convolutions and Fully Connected Layers
    val model1DConvFully = get1DConvFullyModelArchitecture()
    println(model1DConvFully)

    // debugging architecture with dummy Tensor input from above
    // change model for debugging different architectures
    val outputDebug = model1DConvLSTM.updateOutput(inputDebug)
    println("Debugging model architecture Output ...")
    println(outputDebug)
  }

}

