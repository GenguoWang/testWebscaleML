import java.nio.file.{Files, Paths}
import com.intel.webscaleml.nn._
import breeze.linalg.{DenseMatrix=>BDM, *, DenseVector=>BDV}
import scala.math.min
object testMinst {
  type Tensor3D = Array[BDM[Double]]
  var model:Module = null
  var loss:Criterion = null
  var trainData:DataSet = null
  var testData:DataSet = null

  def loadLabel(filename: String): Array[Int] =
  {
    val byteArray = Files.readAllBytes(Paths.get(filename))
    var offset = 4
    var cnt = 0
    for(i<- 0 to 3) cnt = cnt * 256 + (byteArray(offset+i)&0xff)
    offset += 4
    val labels = new Array[Int](cnt)
    for(i<-0 to cnt-1) labels(i) = byteArray(offset+i)&0xff
    println("size: "+cnt)
    //println("label 100,200: " + labels(100) + ", " +labels(200))
    labels
  }

  def loadMatrix(filename: String): Array[BDM[Double]] =
  {
    val byteArray = Files.readAllBytes(Paths.get(filename))
    var offset = 4

    var cnt = 0
    for(i<- 0 to 3) cnt = cnt * 256 + (byteArray(offset+i)&0xff)
    offset += 4

    var rows = 0
    for(i<- 0 to 3) rows = rows * 256 + (byteArray(offset+i)&0xff)
    offset += 4

    var cols = 0
    for(i<- 0 to 3) cols = cols * 256 + (byteArray(offset+i)&0xff)
    offset += 4

    //reshape to 1 row
    cols = rows * cols
    rows = 1

    val matrixs = new Array[BDM[Double]](cnt)
    for(i<-0 to cnt-1)
    {
      matrixs(i) = new BDM[Double](rows,cols)
      for(j<-0 to rows-1)
      {
        for(k<-0 to cols-1)
        {
          matrixs(i)(j,k) = byteArray(offset+i*rows*cols+j*cols+k) & 0xff
        }
      }
    }
    println("size :" + cnt)
    /*
    println("dims: " + rows + " " + cols)
    println("matrix 100:")
    for(j<-0 to rows-1)
    {
      for(k<-0 to cols-1)
      {
        print(matrixs(100)(j,k)+"\t")
      }
      print("\n")
    }
    println("matrix 200:")
    for(j<-0 to rows-1)
    {
      for(k<-0 to cols-1)
      {
        print(matrixs(200)(j,k)+"\t")
      }
      print("\n")
    }
    */
    matrixs
  }

  def loadDataset():(DataSet, DataSet) =
  {
    val train = new DataSet
    val test = new DataSet
    val trainDataPath = "/home/kingo/Workshop/dataset/train-images-idx3-ubyte"
    val trainLabelPath = "/home/kingo/Workshop/dataset/train-labels-idx1-ubyte"
    val testDataPath = "/home/kingo/Workshop/dataset/t10k-images-idx3-ubyte"
    val testLabelPath = "/home/kingo/Workshop/dataset/t10k-labels-idx1-ubyte"
    train.label = loadLabel(trainLabelPath)
    train.data = loadMatrix(trainDataPath)
    train.size = train.label.length
    train.classNumber = 10
    test.label = loadLabel(testLabelPath)
    test.data = loadMatrix(testDataPath)
    test.size = test.label.length
    test.classNumber = 10
    (train,test)
  }

  def getModel(ninputs: Int, noutputs: Int):Module =
  {
    val mlp = new Sequential
    val nhiddens = ninputs/2
    mlp.add(new Linear(ninputs,nhiddens))
    mlp.add(new Tanh)
    mlp.add(new Linear(nhiddens,noutputs))
    mlp.add(new LogSoftMax)
    mlp
  }

  def getCriterion:Criterion =
  {
    new ClassNLLCriterion()
  }

  def train(): Unit =
  {
    val trainData = this.trainData
    val model = this.model
    val loss = this.loss
    val input = new Array[BDM[Double]](1)
    var yCnt = 0
    var err = 0.0
    val startTime = System.nanoTime()
    for(i<-0 to trainData.size - 1)
    {
      //if(i % (trainData.size / 10) == 0) println("training "+i+"/"+trainData.size)
      input(0) = trainData.data(i)
      val label = trainData.label(i)
      val output = model.forward(input).asInstanceOf[Tensor3D]
      if(maxIndex(output(0)) == label) yCnt += 1
      val labelRes = new Array[Double](1)
      labelRes(0) = label
      err = loss.forward(output, labelRes)
      val grad = loss.backward(input,labelRes)
      model.zeroGradParameters()
      model.backward(input,grad)
      model.updateParameters(0.001)
    }
    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    println("train err: "+err)
    val accuracy= 1.0*yCnt/trainData.size
    println("train accuracy: " + accuracy*100 + "%")
    println("train time per sample: " + (elapsedTime*1000.0/trainData.size) + "ms")
  }

  def test(): Unit =
  {
    val testData = this.testData
    val model = this.model
    val loss = this.loss
    val input = new Array[BDM[Double]](1)
    var yCnt = 0
    val startTime = System.nanoTime()
    for(i<-0 to testData.size - 1)
    {
      //if(i % (testData.size / 10) == 0) println("testing "+i+"/"+testData.size)
      input(0) = testData.data(i)
      val label = testData.label(i)
      val output = model.forward(input).asInstanceOf[Tensor3D]
      if(maxIndex(output(0)) == label) yCnt += 1
    }
    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    val accuracy= 1.0*yCnt/testData.size
    println("test accuracy: " + accuracy*100 + "%")
    println("test time per sample: " + (elapsedTime*1000.0/trainData.size) + "ms")
  }

  def maxIndex(data:BDM[Double]):Int = {
    var index = 0
    var maxVal = data(0,0)
    for(j<-1 to data.cols-1)
    {
      if(data(0,j) > maxVal)
      {
        maxVal = data(0,j)
        index = j
      }
    }
    index
  }

  def main (args: Array[String]) {
    val (trainData, testData) = loadDataset()
    //this.trainData, this.testData = loadDataset()
    this.trainData = trainData
    this.testData = testData
    this.model = getModel(28*28,10)
    this.loss = getCriterion
    val maxIter = 10
    for(i<-0 to maxIter-1)
    {
      println("\niterator: " + i)
      train()
      test()
    }
  }
}
