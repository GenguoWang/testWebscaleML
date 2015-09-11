import com.intel.webscaleml.nn._
import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV}
import scala.math.random
import scala.math.log
import scala.math.abs
import scala.math.max
import scala.math.tanh
import scala.math.sqrt

/**
 * Created by kingo on 9/7/15.
 */
object developTesting {
  type Tensor3D = Array[BDM[Double]]
  def debugInfo(info:scala.Any): Unit =
  {
    println(info)
  }
  def testLn: Unit =
  {
    println("Test Linear Module")

    val inputN = 5
    val outputN = 2
    val batchN = 3

    val linear = new Linear(inputN,outputN)
    val mse = new MSECriterion

    //train
    val input = new Array[BDM[Double]](1)
    input(0) = new BDM[Double](batchN,inputN)
    val res = new Array[BDM[Double]](1)
    res(0) = new BDM[Double](batchN,outputN)
    for(i <-1 to 10000)
    {
      input(0).foreachKey((key:(Int,Int)) => input(0)(key) = random)
      res(0).foreachKey((key:(Int,Int)) => if(key._2==0) res(0)(key) = input(0)(key))
      val output = linear.forward(input).asInstanceOf[Tensor3D]
      val err = mse.forward(output, res)
      val grad = mse.backward(output, res).asInstanceOf[Tensor3D]
      linear.zeroGradParameters()
      linear.backward(input,grad)
      linear.updateParameters(0.5/log(i+3))
    }


    //output train result
    input(0).foreachKey((key:(Int,Int)) => input(0)(key) = random)
    res(0).foreachKey((key:(Int,Int)) => if(key._2==0) res(0)(key) = input(0)(key))
    debugInfo("=====bias")
    debugInfo(linear.bias)
    debugInfo("=====weight")
    debugInfo(linear.weight(0))
    debugInfo("=====input")
    debugInfo(input(0))
    debugInfo("=====output")
    val output = linear.forward(input).asInstanceOf[Tensor3D]
    debugInfo(output(0))
    val err = mse.forward(output,res)
    val grad = mse.backward(output,res).asInstanceOf[Tensor3D]
    debugInfo("=====err")
    debugInfo(err)
    debugInfo("=====gradOutput")
    debugInfo(grad(0))
    debugInfo("=====gradInput")
    debugInfo(linear.backward(input,grad).asInstanceOf[Tensor3D](0))
    debugInfo("=====gradWeight")
    debugInfo(linear.gradWeight)
    debugInfo("=====gradBias")
    debugInfo(linear.gradBias)
    val error = (output(0)(0,0)-input(0)(0,0))*(output(0)(0,0)-input(0)(0,0))
    assert(error < 1e-6)
    println("Linear Module Test Passed\n\n")
  }

  def testTanh: Unit =
  {
    println("Test Tanh")

    val inputN = 10
    val hiddenN = inputN/2
    val outputN = 1
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
    var err = 0.0
    for(i <-1 to 1000000)
    {
      input(0).foreachKey((key:(Int,Int)) => input(0)(key) = random)
      res(0).foreachKey((key:(Int,Int)) => if(key._2==0) res(0)(key) = input(0)(key)+100)
      val output = mlp.forward(input).asInstanceOf[Tensor3D]
      err = mse.forward(output, res)
      if(i % (100) == 0) println(err)
      val grad = mse.backward(output, res).asInstanceOf[Tensor3D]
      mlp.zeroGradParameters()
      mlp.backward(input,grad).asInstanceOf[Tensor3D]
      mlp.updateParameters(0.001)
    }

    assert(err < 1e-3)
    println("Tanh Test Passed\n\n")
  }

  def testMSE: Unit =
  {
    println("Test MSE")
    val input = new Array[BDM[Double]](1)
    input(0) = new BDM[Double](3,3)
    input(0)(0,0) = 10
    val target = new Array[BDM[Double]](1)
    target(0) = new BDM[Double](3,3)
    target(0)(0,0) = 8

    val mse = new MSECriterion
    var output = mse.forward(input,target)
    var grad = mse.backward(input,target).asInstanceOf[Tensor3D]
    debugInfo("output1:")
    debugInfo(output)
    debugInfo(grad(0))
    assert(abs(output - 4.0/9) < 1e-6)

    mse.sizeAverage = false
    output = mse.forward(input,target)
    grad = mse.backward(input,target).asInstanceOf[Tensor3D]
    debugInfo("output2:")
    debugInfo(output)
    debugInfo(grad(0))
    assert(abs(output - 4) < 1e-6)
    println("MSE Test Passed\n\n")
  }

  def testTanh_1(): Unit ={
    val tanh = new Tanh
    val input = new Array[BDM[Double]](2)
    input(0) = new BDM[Double](2,2)
    input(0)(0,0) = 1
    input(0)(0,1) = 0.5
    input(0)(1,0) = -0.49
    input(0)(1,1) = 0.28
    input(1) = input(0)
    val output = tanh.forward(input).asInstanceOf[Tensor3D]
    val gradOutput = new Array[BDM[Double]](2)
    gradOutput(0) = BDM.ones[Double](2,2)
    gradOutput(1) = gradOutput(0)
    val gradInput = tanh.backward(input,gradOutput).asInstanceOf[Tensor3D]
    val expectedOutput = new BDM[Double](2,2)
    expectedOutput(0,0) = 0.76159415595576
    expectedOutput(0,1) = 0.46211715726001
    expectedOutput(1,0) = -0.45421643268226
    expectedOutput(1,1) = 0.27290508056313
    val expectedGrad = new BDM[Double](2,2)
    expectedGrad(0,0) = 0.41997434161403
    expectedGrad(0,1) = 0.78644773296593
    expectedGrad(1,0) = 0.7936874322814
    expectedGrad(1,1) = 0.92552281700283

    val delta = output(0) - expectedOutput
    assert(delta.forall( abs(_) < 1e-6 ))

    val deltaGrad = gradInput(0) - expectedGrad
    assert(deltaGrad.forall( abs(_) < 1e-6 ))

    assert(gradOutput(1) == gradOutput(0))
    assert(output(1) == output(0))
    println("testTanh_1 passed")
  }

  def testLogSoftMax(): Unit =
  {
    val logSoftMax = new LogSoftMax
    val input = new Array[BDM[Double]](1)
    input(0) = new BDM[Double](3,3)
    input(0)(0,0) = 1.6416029641405
    input(0)(0,1) = 0.22695825062692
    input(0)(0,2) = -0.30847192369401
    input(0)(1,0) = 1.1675873044878
    input(0)(1,1) = -1.5945216845721
    input(0)(1,2) = 1.9583484353498
    input(0)(2,0) = -0.54572345875204
    input(0)(2,1) = 0.15638273861259
    input(0)(2,2) = -0.56760850828141
    val expectedOutput = new Array[BDM[Double]](1)
    expectedOutput(0) = new BDM[Double](3,3)
    expectedOutput(0)(0,0) = -0.32590548921196
    expectedOutput(0)(0,1) = -1.7405502027255
    expectedOutput(0)(0,2) = -2.2759803770464
    expectedOutput(0)(1,0) = -1.1842649262777
    expectedOutput(0)(1,1) = -3.9463739153377
    expectedOutput(0)(1,2) = -0.39350379541579
    expectedOutput(0)(2,0) = -1.3853819245743
    expectedOutput(0)(2,1) = -0.68327572720965
    expectedOutput(0)(2,2) = -1.4072669741036
    val expectedGrad = new Array[BDM[Double]](1)
    expectedGrad(0) = new BDM[Double](3,3)
    expectedGrad(0)(0,0) = -1.1656202270977
    expectedGrad(0)(0,1) = 0.47372843387093
    expectedGrad(0)(0,2) = 0.69191146572779
    expectedGrad(0)(1,0) = 0.082086975689136
    expectedGrad(0)(1,1) = 0.94202605692268
    expectedGrad(0)(1,2) = -1.0240662705429
    expectedGrad(0)(2,0) = 0.24931536029314
    expectedGrad(0)(2,1) = -0.51488050529594
    expectedGrad(0)(2,2) = 0.26556566288322
    val output = logSoftMax.forward(input).asInstanceOf[Tensor3D]
    val gradOutput = new Array[BDM[Double]](1)
    gradOutput(0) = BDM.ones[Double](3,3)
    val gradInput = logSoftMax.backward(input,gradOutput).asInstanceOf[Tensor3D]
    assert((output(0) - expectedOutput(0)).forall( abs(_) < 1e-4 ))
    assert((gradInput(0) - expectedGrad(0)).forall( abs(_) < 1e-4 ))
    println("testLogSoftMax passed")
  }

  def testClassNLLCriterion(): Unit =
  {
    var classNll = new ClassNLLCriterion()
    val input = new Array[BDM[Double]](1)
    input(0) = new BDM[Double](3,4)
    input(0)(0,0) = -13.068250889195
    input(0)(0,1) = -14.667719133435
    input(0)(0,2) = -2.7243095741176
    input(0)(0,3) = -0.067855884557344
    input(0)(1,0) = -0.0050048546048291
    input(0)(1,1) = -19.36802823256
    input(0)(1,2) = -5.3047696191
    input(0)(1,3) = -14.211689840734
    input(0)(2,0) = -6.7004034256968e-05
    input(0)(2,1) = -14.688393580759
    input(0)(2,2) = -9.82249596625
    input(0)(2,3) = -11.762182205016
    val target = new Array[Double](3)
    target(0) = 0
    target(1) = 1
    target(2) = 2
    var expectedOutput = 14.086258362668

    val expectedGrad = new Array[BDM[Double]](1)
    expectedGrad(0) = new BDM[Double](3,4)
    expectedGrad(0)(0,0) = -0.33333333333333
    expectedGrad(0)(0,1) = 0
    expectedGrad(0)(0,2) = 0
    expectedGrad(0)(0,3) = 0
    expectedGrad(0)(1,0) = 0
    expectedGrad(0)(1,1) = -0.33333333333333
    expectedGrad(0)(1,2) = 0
    expectedGrad(0)(1,3) = 0
    expectedGrad(0)(2,0) = 0
    expectedGrad(0)(2,1) = 0
    expectedGrad(0)(2,2) = -0.33333333333333
    expectedGrad(0)(2,3) = 0

    var output = classNll.forward(input,target)
    var gradInput = classNll.backward(input,target).asInstanceOf[Tensor3D]
    assert(abs(output-expectedOutput) < 1e-6)
    assert((gradInput(0) - expectedGrad(0)).forall( abs(_) < 1e-6 ))

    //with weights
    val weights = new Array[Double](3)
    weights(0) = 0.3
    weights(1) = 0.33
    weights(2) = 0.37
    expectedOutput = 4.6487493636719 * 3
    expectedGrad(0)(0,0) = -0.1
    expectedGrad(0)(0,1) = 0
    expectedGrad(0)(0,2) = 0
    expectedGrad(0)(0,3) = 0
    expectedGrad(0)(1,0) = 0
    expectedGrad(0)(1,1) = -0.11
    expectedGrad(0)(1,2) = 0
    expectedGrad(0)(1,3) = 0
    expectedGrad(0)(2,0) = 0
    expectedGrad(0)(2,1) = 0
    expectedGrad(0)(2,2) = -0.12333333333333
    expectedGrad(0)(2,3) = 0
    expectedGrad(0) :*= 3.0
    classNll = new ClassNLLCriterion(weights)
    output = classNll.forward(input,target)
    println(output)
    gradInput = classNll.backward(input,target).asInstanceOf[Tensor3D]
    assert(abs(output-expectedOutput) < 1e-6)
    assert((gradInput(0) - expectedGrad(0)).forall( abs(_) < 1e-6 ))
    print("testClassNLLCriterion passed")
  }
  def main(args: Array[String]): Unit = {
    testLn
    testMSE
    testTanh
    testTanh_1
    testLogSoftMax()
    testClassNLLCriterion()
  }
}

