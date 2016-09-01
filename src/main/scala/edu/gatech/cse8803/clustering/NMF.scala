
package edu.gatech.cse8803.clustering

/**
  * @author Ting Pan <tpan35@gatech.edu>
  */


import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, sum}
import breeze.linalg._
import breeze.numerics._
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec.B
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Matrices


object NMF {

  /**
    * Run NMF clustering
    *
    * @param V The original non-negative matrix
    * @param k The number of clusters to be formed, also the number of cols in W and number of rows in H
    * @param maxIterations The maximum number of iterations to perform
    * @param convergenceTol The maximum change in error at which convergence occurs.
    * @return two matrixes W and H in RowMatrix and DenseMatrix format respectively
    */
  def run(V: RowMatrix, k: Int, maxIterations: Int, convergenceTol: Double = 1e-4): (RowMatrix, BDM[Double]) = {

    /**
      * 
      * Initialize W, H randomly
      * Calculate the initial error (Euclidean distance between V and W * H)
      */
    var W = new RowMatrix(V.rows.map(_ => BDV.rand[Double](k)).map(fromBreeze).cache)
    var H = BDM.rand[Double](k, V.numCols().toInt)

    val WH = multiply(W,H)
    val InialE = error(V,WH)
    println("Initial Eorror: "+InialE)
    /**
      * 
      * Iteratively update W, H in a parallel fashion until error falls below the tolerance value
      * The updating equations are,
      * H = H.* W^T^V ./ (W^T^W H+10e-9)
      * W = W.* VH^T^ ./ (W H H^T^+10e-9)
      */
    var prevError = 0.0
    var delta = 0.0
    var iterations = 0
    println("                        ")
    do {
      prevError = delta

      val WtV = computeWTV(W,V)
      val Ws = computeWTV(W,W)
      val WsH = Ws*H

      val div= WtV:/(WsH:+10.0e-9)
      /** update H using old W*/
      H = H:*div


      val VHt = multiply(V,H.t)
      val Hs = H*H.t
      val WtHs = multiply(W,Hs)
      val preW=dotDiv(VHt,WtHs)
      /** update W using new H*/
      W=dotProd(W,preW)
      W.rows.cache.count

      val newMatrix = multiply(W,H)
      delta = error(V, newMatrix)
      println("Iterations: "+iterations+", "+"current error: "+delta)
      iterations += 1
    } while (iterations < maxIterations && delta > convergenceTol)

    (W, H)
  }

  private def error(a:RowMatrix,b:RowMatrix):Double={
    val diff = new RowMatrix(a.rows.zip(b.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :- toBreezeVector(v2)
    }.map(fromBreeze))
    val dist = dotProd(diff,diff).rows.map(x=> x.toArray.sum).reduce(_+_)
    dist*0.5
  }


  /**
    * 
    * recommended helper functions for matrix manipulation
    * For the implementation of the first three helper functions (with a null return),
    * you can refer to dotProd and dotDiv whose implementation are provided
    */
  /**
    * Note:You can find some helper functions to convert vectors and matrices
    * from breeze library to mllib library and vice versa in package.scala
    */

  /** compute the mutiplication of a RowMatrix and a dense matrix
    * add private back*/
  private def multiply(X: RowMatrix, d: BDM[Double]): RowMatrix = {
    /**val res = getDenseMatrix(X)*d */
    val res = X.multiply(fromBreeze_new(d))
    res
  }

  /** get the dense matrix representation for a RowMatrix
    * add private back*/
  private def getDenseMatrix(X: RowMatrix): BDM[Double] = {
    /**
    val n=X.numRows()
   val m=X.numCols()
   val Xdense = X.rows.map{v=>
     Matrices.dense(m.toInt,m.toInt,v.toArray)}
   val res = Xdense.map(x=> toBreezeMatrix(x)).asInstanceOf[BDM[Double]]
   res
      */
    null
  }

  /** matrix multiplication of W.t and V */
  def computeWTV(W: RowMatrix, V: RowMatrix): BDM[Double] = {
    val S = W.rows.zip(V.rows).map{case (v1: Vector, v2: Vector) =>
      def OuterProd(v1:Vector, v2:Vector):Array[Double]={
        val prod=v2.toArray.flatMap(r=>v1.toArray.map(v=>v*r))
        prod
      }
      val newArray = OuterProd(v1, v2)
      Matrices.dense(W.numCols().toInt, V.numCols().toInt, newArray)}
    val WTV=S.map(r => toBreezeMatrix(r)).reduce(_+_).asInstanceOf[BDM[Double]]
    WTV
  }



  /** elementwise dot product of two RowMatrixes */
  def dotProd(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :* toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  /** elementwise dot division of two RowMatrixes */
  def dotDiv(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :/ toBreezeVector(v2).mapValues(_ + 10.0e-9)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  def nonZero(X:RowMatrix, Y:Double):RowMatrix={
    val rows = X.rows.map { case (v: Vector) =>
      toBreezeVector(v) + Y
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

}
