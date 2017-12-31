package incertum.cybermatrixai.url

import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import com.intel.analytics.bigdl.utils.Engine

trait SparkInit {

  val conf = Engine.createSparkConf()
    .setAppName("Spark Scala BigDL Deep Learning URL Classification")
    .setMaster("local[6]") // BigDL requires the batch size to be a multiple of nodeNumber * coreNumber
    .set("spark.task.maxFailures", "1")
    .set("spark.executor.memory", "6G")
    .set("spark.driver.memory", "8G")
    .set("spark.driver.maxResultSize", "10G")

  val sc = new SparkContext(conf)

  val sqlContext = new SQLContext(sc)
  val spark: SparkSession = sqlContext.sparkSession

  Engine.init

  private def init = {
    sc.setLogLevel("ERROR")
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    LogManager.getRootLogger.setLevel(Level.ERROR)
  }

  init

  def close = {
    spark.stop()
  }

}
