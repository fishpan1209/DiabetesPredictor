/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse8803.clustering

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

object Metrics {
  /**
   * Given input RDD with tuples of assigned cluster id by clustering,
   * and corresponding real class. Calculate the purity of clustering.
   * Purity is defined as
   *             \fract{1}{N}\sum_K max_j |w_k \cap c_j|
   * where N is the number of samples, K is number of clusters and j
   * is index of class. w_k denotes the set of samples in k-th cluster
   * and c_j denotes set of samples of class j.
   * @param clusterAssignmentAndLabel RDD in the tuple format
   *                                  (assigned_cluster_id, class)
   * @return purity
   */
  def purity(clusterAssignmentAndLabel: RDD[(Int, Int)]): Double = {
    /**
     * TODO: Remove the placeholder and implement your code here
     */
    val N=clusterAssignmentAndLabel.cache().count()
    val mycluster=clusterAssignmentAndLabel.map(_._1).distinct.collect()
    val myclass=clusterAssignmentAndLabel.map(_._2).distinct.collect()
    val k=mycluster.length
    val j=myclass.length
    var sum=0.0;
    /** for each cluster*/
    for(i<- 0 to k-1){
      val cluster_id=mycluster(i)
      var max=0;
      for(j<- 0 to j-1){
        val class_id=myclass(j)
        val union=clusterAssignmentAndLabel.filter(t=>((t._1==cluster_id)&&(t._2==class_id))).count()
        if(max<=union){
          max=union.toInt
        }
      }
      sum=sum+max;
    }
    val result=sum/(N.toDouble)
   result
  }
  def confusionMatrix(clusterAssignmentAndLabel: RDD[(Int, Int)]) = {
    val N = clusterAssignmentAndLabel.count().toDouble
    val K = clusterAssignmentAndLabel.map(_._2).distinct().count().toInt
    for (i <- 1 until K+1) {
      var sum = 0.0;
      val item = clusterAssignmentAndLabel.filter(_._2 == i).cache()
      val cluster1 = item.filter(_._1 == 0).count().toDouble
      val cluster2 = item.filter(_._1 == 1).count().toDouble
      val cluster3 = item.filter(_._1 == 2).count().toDouble
      sum = cluster1 + cluster2 + cluster3

      println(f"Patient group: " + i + " Cluster 1: " + cluster1/sum + " Cluster 2: " + cluster2/sum + " Cluster3: " + cluster3/sum )
    }
  }

}
