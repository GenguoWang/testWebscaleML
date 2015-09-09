/**
 * Created by kingo on 9/8/15.
 */

import scala.language.reflectiveCalls
import breeze.linalg.{DenseMatrix=>BDM, *, DenseVector=>BDV}
import com.intel.webscaleml.nn._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.sql.types.StringType
import org.apache.spark.mllib.linalg.Vector
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.math._

class DataSet
{
  var data:Array[BDM[Double]] = null
  var label:Array[Int] = null
  var size:Int = 0
}

object testNN {

   def loadData(
                            sc: SparkContext,
                            path: String,
                            format: String,
                            expectedNumFeatures: Option[Int] = None): RDD[LabeledPoint] = {
    format match {
      case "dense" => MLUtils.loadLabeledPoints(sc, path)
      case "libsvm" => expectedNumFeatures match {
        case Some(numFeatures) => MLUtils.loadLibSVMFile(sc, path, numFeatures)
        case None => MLUtils.loadLibSVMFile(sc, path)
      }
      case _ => throw new IllegalArgumentException(s"Bad data format: $format")
    }
  }

  /**
   * Load training and test data from files.
   * @param input  Path to input dataset.
   * @param dataFormat  "libsvm" or "dense"
   * @param testInput  Path to test dataset.
   * @param algo  Classification or Regression
   * @param fracTest  Fraction of input data to hold out for testing.  Ignored if testInput given.
   * @return  (training dataset, test dataset)
   */
   def loadDatasets(
                                sc: SparkContext,
                                input: String,
                                dataFormat: String,
                                testInput: String,
                                algo: String,
                                fracTest: Double): (DataFrame, DataFrame) = {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    // Load training data
    val origExamples: RDD[LabeledPoint] = loadData(sc, input, dataFormat)

    // Load or create test set
    val splits: Array[RDD[LabeledPoint]] = if (testInput != "") {
      // Load testInput.
      val numFeatures = origExamples.take(1)(0).features.size
      val origTestExamples: RDD[LabeledPoint] =
        loadData(sc, testInput, dataFormat, Some(numFeatures))
      Array(origExamples, origTestExamples)
    } else {
      // Split input into training, test.
      origExamples.randomSplit(Array(1.0 - fracTest, fracTest), seed = System.nanoTime())
    }

    // For classification, convert labels to Strings since we will index them later with
    // StringIndexer.
    def labelsToStrings(data: DataFrame): DataFrame = {
      algo.toLowerCase match {
        case "classification" =>
          data.withColumn("labelString", data("label").cast(StringType))
        case "regression" =>
          data
        case _ =>
          throw new IllegalArgumentException("Algo ${params.algo} not supported.")
      }
    }
    val dataframes = splits.map(_.toDF()).map(labelsToStrings)
    val training = dataframes(0).cache()
    val test = dataframes(1).cache()

    val numTraining = training.count()
    val numTest = test.count()
    val numFeatures = training.select("features").first().getAs[Vector](0).size
    println("Loaded data:")
    println(s"  numTraining = $numTraining, numTest = $numTest")
    println(s"  numFeatures = $numFeatures")

    (training, test)
  }

  def main(args:Array[String]): Unit =
  {
    println("hello test")
    val conf = new SparkConf().setAppName("test ML").setMaster("local")
    val sc = new SparkContext(conf)
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)


    // Load training and test data and cache it.
    val inputFile: String = "/home/kingo/Workshop/spark-1.4.1-bin-hadoop2.4/data/mllib/sample_libsvm_data.txt"
    val testInput: String = ""
    val dataFormat: String = "libsvm"
    val maxIter: Int = 50
    val fracTest: Double = 0.3

    //val test: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc,input).toDF()
    val (training: DataFrame, test: DataFrame) = loadDatasets(sc, inputFile,
      dataFormat, testInput, "classification", fracTest)
    val testDs = getDataSet(test)
    val trainDs = getDataSet(training)
    println("labelArray(9)",testDs.label(9))
    println("dataArray(9)",testDs.label(9))

    val inputN = testDs.data(0).cols
    println("inputN", inputN)
    val hiddenN = inputN/2
    val outputN = 2
    val batchN = 1

    //create model
    val mlp = new Sequential
    mlp.add(new Linear(inputN, hiddenN))
    mlp.add(new Tanh)
    mlp.add(new Linear(hiddenN,outputN))
    val mse = new MSECriterion

    val input = new Array[BDM[Double]](1)
    input(0) = new BDM[Double](batchN,inputN)
    val res = new Array[BDM[Double]](1)
    res(0) = new BDM[Double](batchN,outputN)
    var j=0
    var err = 0.0
    while(j<maxIter)
    {
      println("iter " +j +": ")
      var i = 0
      var yCnt = 0

      var startTime = System.nanoTime()
      while(i < trainDs.size)
      {
        input(0) = trainDs.data(i)
        val label = trainDs.label(i)
        res(0)(0,label) = 1
        res(0)(0,1-label) = 0
        val output = mlp.forward(input)
        if(output(0)(0,label)>output(0)(0,1-label)) yCnt += 1
        err = mse.forward(output, res)
        val grad = mse.backward(output, res)
        mlp.backward(input,grad)
        mlp.updateParameters(0.001)
        i += 1
      }
      var elapsedTime = (System.nanoTime() - startTime) / 1e9
      println("train err",err)
      println("train accuracy",1.0*yCnt/trainDs.size)
      println("train time",elapsedTime)
      i = 0
      yCnt = 0
      startTime = System.nanoTime()
      while(i < testDs.size)
      {
        input(0) = testDs.data(i)
        val label = testDs.label(i)
        val output = mlp.forward(input)
        if(output(0)(0,label)>output(0)(0,1-label)) yCnt += 1
        i += 1
      }
      elapsedTime = (System.nanoTime() - startTime) / 1e9
      println("test accuracy",1.0*yCnt/testDs.size)
      println("test time",elapsedTime)
      j += 1
    }

    println("Tanh Test Passed\n\n")

    sc.stop()
  }
  def getDataSet(df: DataFrame):DataSet ={
    val row = df.first()
    val dataSize = df.count().toInt
    val featureNum = row.getAs[SparseVector](1).size
    println("dataSize", dataSize)
    println("featureNum", featureNum)
    val dataArray = new Array[BDM[Double]](dataSize)
    val labelArray = new Array[Int](dataSize)
    var i=0
    df.collect().foreach(row => {
      val labelD = row.getDouble(0)
      val label:Int = labelD.toInt
      val labelString:String = row.getString(2)
      assert(abs(label-labelD) < 1e-9)
      assert(abs(label-labelString.toDouble.toInt) < 1e-9)
      assert(label==0 || label==1)
      labelArray(i) = label
      val dataA = row.getAs[SparseVector](1).toArray
      val data = new BDV[Double](dataA)
      dataArray(i) = new BDM[Double](1,featureNum)
      dataArray(i)(*, ::) := data
      var sum = 0.0
      dataArray(i).foreachValue(sum += _)
      println("label",label)
      println("sum", sum)
      i += 1
    })
    val dataSet = new DataSet
    dataSet.data = dataArray
    dataSet.label = labelArray
    dataSet.size = dataSize
    dataSet
  }
}
