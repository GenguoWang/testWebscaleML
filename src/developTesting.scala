package com.intel.webscaleml.nn
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
    var i = 0
    while(i < 10000)
    {
      input(0).foreachKey((key:(Int,Int)) => input(0)(key) = random)
      res(0).foreachKey((key:(Int,Int)) => if(key._2==0) res(0)(key) = input(0)(key))
      val output = linear.forward(input)
      val err = mse.forward(output, res)
      val grad = mse.backward(output, res)
      linear.backward(input,grad)
      linear.updateParameters(0.5/log(i+3))
      i += 1
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
    val output = linear.forward(input)
    debugInfo(output(0))
    val err = mse.forward(output,res)
    val grad = mse.backward(output,res)
    debugInfo("=====err")
    debugInfo(err)
    debugInfo("=====gradOutput")
    debugInfo(grad(0))
    debugInfo("=====gradInput")
    debugInfo(linear.backward(input,grad)(0))
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
    var i = 0
    var err = 0.0
    while(i < 1000000)
    {
      input(0).foreachKey((key:(Int,Int)) => input(0)(key) = random)
      res(0).foreachKey((key:(Int,Int)) => if(key._2==0) res(0)(key) = input(0)(key)+100)
      val output = mlp.forward(input)
      err = mse.forward(output, res)
      if(i % (100) == 0) println(err)
      val grad = mse.backward(output, res)
      mlp.backward(input,grad)
      mlp.updateParameters(0.001)
      i += 1
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
    var grad = mse.backward(input,target)
    debugInfo("output1:")
    debugInfo(output)
    debugInfo(grad(0))
    assert(abs(output - 4) < 1e-6)

    mse.sizeAverage = true
    output = mse.forward(input,target)
    grad = mse.backward(input,target)
    debugInfo("output2:")
    debugInfo(output)
    debugInfo(grad(0))
    assert(abs(output - 4/3.0) < 1e-6)
    println("MSE Test Passed\n\n")
  }
  def main(args: Array[String]): Unit = {
    testLn
    testMSE
    testTanh
  }
}

