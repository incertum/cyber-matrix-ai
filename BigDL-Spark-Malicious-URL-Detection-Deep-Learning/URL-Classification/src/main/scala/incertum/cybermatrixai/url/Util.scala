package incertum.cybermatrixai.url

import Driver._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.UserDefinedFunction
import collection.mutable.HashMap

/**
  * Contributors: incertum (December, 2017)
  */

object Util {

  def loadUrlData(fname: String): DataFrame = {

    var df = spark.read
      .format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .option("delimiter", ",")
      .load(fname)
    return df

  }

  def padUrlIndices(input: Seq[Float]): Seq[Float] = {

    val sequenceLen: Int = 75
    var paddedSeq = input

    while (paddedSeq.size < sequenceLen) {
      paddedSeq = paddedSeq :+ 1.0f

    }

    return paddedSeq

  }

  val padUrlIndices_udf: UserDefinedFunction = udf(padUrlIndices _)

  def getVocabularyHashMap(): HashMap[String, Float] = {
    // For clarity and simplicity typed all included characters, can and should be extended on ...
    val keys = List("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i",
      "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
      "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
      "U", "V", "W", "X", "Y", "Z", "!", "#", "$", "%", "&", "(", ")", "*", "+", ",", "-", ".", "/", ":",
      ";", "<", "=", ">", "?", "@", "[", "]", "^", "_")

    val values = List.range(1, keys.size).map(_.toFloat)
    val m = HashMap(keys.zip(values).toArray: _*)
    return m

  }

}