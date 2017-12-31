package incertum.cybermatrixai.url

import Util.{getVocabularyHashMap, loadUrlData, padUrlIndices_udf}
import incertum.cybermatrixai.url.DeepLearning

import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions._

/**
  * Contributors: incertum (December, 2017)
  */

object Driver extends SparkInit {

  def main(args: Array[String]) {
    import spark.implicits._

    println("Starting BigDL Deep Learning Malicious URL Detection Analytics")

//    val dataPath = "src/main/resources/url_data_sample_deep_learning.csv"
    val dataPath = "../data/url_data_full_deep_learning.csv"
    println(s"Loading following path ${dataPath}")

    var df = loadUrlData(dataPath)
    df.sample(withReplacement = false, fraction = 0.1).show()
    df.printSchema

    /*
     Preprocessing of raw URLs
      */

    // Map each character of URL to Index of Vocabulary Map via resolving broadcast variable
    val sequenceLen = 75
    val vocab = getVocabularyHashMap()
    val vocabBC = sc.broadcast(vocab)
    val resolveBC_udf = udf((urlCharArray: Seq[String]) => urlCharArray.take(sequenceLen)
      .map(c => vocabBC.value.getOrElse(c, 2.0f) + 2.0f))

    // apply functions to df columns
    df = df.withColumn("urlLength", length($"url"))
      .withColumn("url", regexp_replace($"url", "[\\s+]", ""))
      .withColumn("urlTokens", split($"url", ""))
      .withColumn("urlIndices", padUrlIndices_udf(resolveBC_udf($"urlTokens")))
      .withColumn("urlIndicesLength", size($"urlIndices"))

    df.show()

    /*
    Deep Learning BigDL
     */

    val start = System.currentTimeMillis

    val dl = new DeepLearning()

    println("!! %% Debugging BigDL Architectures")
    dl.debugArchitectures()

    println("!! %% Training Deep Learning")
    dl.trainModels(df)


    val totalTime = System.currentTimeMillis - start

    println("Deep Learning BigDL Elapsed time: %1d min and %1d sec".
      format((totalTime / (1000 * 60)) % 60, (totalTime / 1000) % 60))


  }

}
