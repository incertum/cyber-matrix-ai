name := "URL-Classification"

version := "0.1"

scalaVersion := "2.11.8"

resolvers += "Typesafe Repo" at "http://repo.typesafe.com/typesafe/releases/"
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

val repo = "http://repo1.maven.org/maven2"
def mkl_native(os: String): String = {
  s"${repo}/com/intel/analytics/bigdl/native/mkl-java-${os}/0.3.0/mkl-java-${os}-0.3.0.jar"
}

def bigquant_native(os: String): String = {
  s"${repo}/com/intel/analytics/bigdl/bigquant/bigquant-java-${os}/0.3.0/bigquant-java-${os}-0.3.0.jar"

}

val sparkVer = "2.2.0"

//libraryDependencies ++= {
//  Seq(
//    "com.typesafe.play" %% "play-json" % "2.3.8",
//    "org.apache.spark" %% "spark-core" % sparkVer,
//    "org.apache.spark" %% "spark-mllib" % sparkVer,
//    "org.apache.spark" %% "spark-sql" % sparkVer,
//    "org.apache.spark" %% "spark-streaming" % sparkVer
//  )
//}

libraryDependencies ++= {
  Seq(
    "com.typesafe.play" %% "play-json" % "2.3.8",
    "org.apache.spark" %% "spark-core" % sparkVer % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVer % "provided",
    "org.apache.spark" %% "spark-sql" % sparkVer % "provided",
    "org.apache.spark" %% "spark-streaming" % sparkVer % "provided"
  )
}

// Intel's BigDL for Deep Learning on Spark
// https://bigdl-project.github.io/master/#ScalaUserGuide/install-pre-built/

val bigDealVer = "0.3.0"

libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-SPARK_2.2" % bigDealVer exclude("com.intel.analytics.bigdl", "bigdl-core")
libraryDependencies += "com.intel.analytics.bigdl.native" % "mkl-java-mac" % bigDealVer from mkl_native("mac")
libraryDependencies += "com.intel.analytics.bigdl.bigquant" % "bigquant-java-mac" % bigDealVer from bigquant_native("mac")

// Linux
libraryDependencies += "com.intel.analytics.bigdl.native" % "mkl-java" % bigDealVer
libraryDependencies += "com.intel.analytics.bigdl.bigquant" % "bigquant-java" % bigDealVer

assemblyMergeStrategy in assembly := {
  case PathList("javax", "servlet", xs@_*) => MergeStrategy.last
  case PathList("javax", "activation", xs@_*) => MergeStrategy.last
  case PathList("javax", xs@_*) => MergeStrategy.last
  case PathList("org", "apache", xs@_*) => MergeStrategy.last
  case PathList("com", "google", xs@_*) => MergeStrategy.last
  case PathList("com", "googlecode", xs@_*) => MergeStrategy.last
  case PathList("com", "esotericsoftware", xs@_*) => MergeStrategy.last
  case PathList("com", "codahale", xs@_*) => MergeStrategy.last
  case PathList("com", "yammer", xs@_*) => MergeStrategy.last
  case PathList("org", "aopalliance", xs@_*) => MergeStrategy.last
  case PathList(ps@_*) if ps.last endsWith ".html" => MergeStrategy.last
  case "about.html" => MergeStrategy.rename
  case "META-INF/ECLIPSEF.RSA" => MergeStrategy.last
  case "META-INF/mailcap" => MergeStrategy.last
  case "META-INF/mimetypes.default" => MergeStrategy.last
  case "plugin.properties" => MergeStrategy.last
  case "log4j.properties" => MergeStrategy.last
  case x if x.contains("com/intel/analytics/bigdl/bigquant/") => MergeStrategy.first // BigDL
  case x if x.contains("com/intel/analytics/bigdl/mkl/") => MergeStrategy.first //BigDL
  case x => (assemblyMergeStrategy in assembly).value(x)
}

