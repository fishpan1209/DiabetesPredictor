/**
 * @author Ting Pan
 */
package edu.gatech.cse8803.features

import edu.gatech.cse8803.model.{LabResult, Medication, Diagnostic}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._


object FeatureConstruction {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String), Double)

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
    *
    * @param diagnostic RDD of diagnostic
   * @return RDD of feature tuples
    * For diagnostic and medication features, you must count the number of times a code or a medication appears for a given patient. That will be your feature value.
For lab result, you must average the values for a given test for a given patient. That will be your feature value
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]): RDD[FeatureTuple] = {
    
    val diag = diagnostic.map(x=>(x.patientID,x.code)).countByValue().map(x=>(x._1,x._2.toDouble))
    diagnostic.sparkContext.parallelize(diag.toSeq)
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation,
    *
    * @param medication RDD of medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {
    
    val med = medication.map(x=>(x.patientID,x.medicine))
      .countByValue()
      .map(x=>(x._1,x._2.toDouble))
    medication.sparkContext.parallelize(med.toSeq)
  }

  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
    *
    * @param labResult RDD of lab result
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult]): RDD[FeatureTuple] = {
    
    val lab = labResult.map{x=>
      val feature = (x.patientID,x.testName)
      (feature,(x.value))
    }.groupByKey().map(x=>(x._1,x._2.sum/x._2.size))

    labResult.sparkContext.parallelize(lab.collectAsMap().toSeq)
  }

  /**
   * Aggregate feature tuple from diagnostics with COUNT aggregation, but use code that is
   * available in the given set only and drop all others.
    *
    * @param diagnostic RDD of diagnostics
   * @param candiateCode set of candidate code, filter diagnostics based on this set
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candiateCode: Set[String]): RDD[FeatureTuple] = {
    
    val diag = diagnostic.filter(d => candiateCode.contains(d.code))
      .map(x=>(x.patientID,x.code))
      .countByValue()
      .map(x=>(x._1,x._2.toDouble))
    diagnostic.sparkContext.parallelize(diag.toSeq)
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation, use medications from
   * given set only and drop all others.
    *
    * @param medication RDD of diagnostics
   * @param candidateMedication set of candidate medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set[String]): RDD[FeatureTuple] = {
    
    val med = medication.filter(m=>candidateMedication.contains(m.medicine))
      .map(x=>(x.patientID,x.medicine))
        .countByValue()
        .map(x=>(x._1,x._2.toDouble))
    medication.sparkContext.parallelize(med.toSeq)
  }


  /**
   * Aggregate feature tuples from lab result with AVERAGE aggregation, use lab from
   * given set of lab test names only and drop all others.
    *
    * @param labResult RDD of lab result
   * @param candidateLab set of candidate lab test name
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set[String]): RDD[FeatureTuple] = {
   
    val lab = labResult.filter(l=>candidateLab.contains(l.testName))
      .map{x=>
      val feature = (x.patientID,x.testName)
      (feature,(x.value))
    }.groupByKey()
      .map(x=>(x._1,x._2.sum/x._2.size))

    labResult.sparkContext.parallelize(lab.collectAsMap().toSeq)
  }


  /**
   * Given a feature tuples RDD, construct features in vector
   * format for each patient. feature name should be mapped
   * to some index and convert to sparse feature format.
    *
    * @param sc SparkContext to run
   * @param feature RDD of input feature tuples
   * @return
   */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    /** create a feature name to id map*/


    /** transform input feature
      * (patientID, featureVector)*/

    /**
     * Functions maybe helpful:
     *    collect
     *    groupByKey
     */

    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */


    val grpFeature = feature.map{x=>
      val patientID =x._1._1
    val features = (x._1._2,x._2)
    (patientID, features)}.groupByKey().map{case(patientID, features)=>
    val featureList = features.toArray.toMap
      (patientID, featureList)}

    val featureMap = grpFeature.flatMap(_._2.keys).distinct.collect.zipWithIndex.toMap
    val scFeatureMap = sc.broadcast(featureMap)
   println("feature size    "+scFeatureMap.value.size)
    val finalFeature = grpFeature.map{case(patientID, features)=>
      val numFeature = scFeatureMap.value.size
      val indexedFeatures = features.toList.map{case(featureName, featureValue)=>(scFeatureMap.value(featureName),featureValue)}
    val featureVector = Vectors.sparse(numFeature, indexedFeatures)
      (patientID, featureVector)}


    val result = finalFeature
    result
    /** The feature vectors returned can be sparse or dense. It is advisable to use sparse */

  }
}
