/**
 * @author Ting Pan <tpan35@gatech.edu>.
 */

package edu.gatech.cse8803.main

import java.text.SimpleDateFormat

import edu.gatech.cse8803.clustering.{NMF, Metrics}
import edu.gatech.cse8803.features.FeatureConstruction
import edu.gatech.cse8803.ioutils.CSVUtils
import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import edu.gatech.cse8803.phenotyping.T2dmPhenotype
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Vectors, Vector}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source


object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.Logger
    import org.apache.log4j.Level

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = createContext
    val sqlContext = new SQLContext(sc)
    /** initialize loading of data */
    val (medication, labResult, diagnostic) = loadRddRawData(sqlContext)
    val (candidateMedication, candidateLab, candidateDiagnostic) = loadLocalRawData

    /** conduct phenotyping */
    val phenotypeLabel = T2dmPhenotype.transform(medication, labResult, diagnostic)

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult),
      FeatureConstruction.constructMedicationFeatureTuple(medication)
    )

    val rawFeatures = FeatureConstruction.construct(sc, featureTuples).cache()



    val (kMeansPurity, gaussianMixturePurity, nmfPurity) = testClustering(phenotypeLabel, rawFeatures)
    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    println(f"[All feature] purity of NMF is: $nmfPurity%.5f")

    /** feature construction with filtered features */
    val filteredFeatureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic, candidateDiagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult, candidateLab),
      FeatureConstruction.constructMedicationFeatureTuple(medication, candidateMedication)
    )

    val filteredRawFeatures = FeatureConstruction.construct(sc, filteredFeatureTuples).cache()



    val (kMeansPurity2, gaussianMixturePurity2, nmfPurity2) = testClustering(phenotypeLabel, filteredRawFeatures)
    println(f"[Filtered feature] purity of kMeans is: $kMeansPurity2%.5f")
    println(f"[Filtered feature] purity of GMM is: $gaussianMixturePurity2%.5f")
    println(f"[Filtered feature] purity of NMF is: $nmfPurity2%.5f")


    sc.stop
  }

  def testClustering(phenotypeLabel: RDD[(String, Int)], rawFeatures:RDD[(String, Vector)]): (Double, Double, Double) = {
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix

    /** scale features */
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(rawFeatures.map(_._2))
    val features = rawFeatures.map({ case (patientID, featureVector) => (patientID, scaler.transform(Vectors.dense(featureVector.toArray)))})
    val rawFeatureVectors = features.map(_._2).cache()

    /** reduce dimension */
    val mat: RowMatrix = new RowMatrix(rawFeatureVectors)
    val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
    val featureVectors = mat.multiply(pc).rows

    val densePc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray).asInstanceOf[DenseMatrix]
    /** transform a feature into its reduced dimension representation */
    def transform(feature: Vector): Vector = {
      Vectors.dense(Matrices.dense(1, feature.size, feature.toArray).multiply(densePc).toArray)
    }

    /** TODO: K Means Clustering using spark mllib
      *  Train a k means model using the variabe featureVectors as input
      *  Set maxIterations =20 and seed as 0L
      *  Assign each feature vector to a cluster(predicted Class)
      *  Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
      *  Find Purity using that RDD as an input to Metrics.purity
      *  Remove the placeholder below after your implementation
      */
    import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
    import org.apache.spark.mllib.linalg.Vectors
    val sc = rawFeatures.sparkContext
    val kmm=KMeans.train(featureVectors, k=3, maxIterations=20,1,"k-means||",0L)
    /**
    val predKMM = kmm.predict(featureVectors)
    val assignKMM = features.map(_._1).collect().zip(predKMM.collect())
    val assignmentKMM = sc.parallelize(assignKMM).map(_._2)
    val labelKMM = features.join(phenotypeLabel).map(_._2._2)
    val kmmClusterAssignmentAndLabel=assignmentKMM.zipWithIndex().map(_.swap).join(labelKMM.zipWithIndex().map(_.swap)).map(_._2)*/
    val kmmClusterAssignmentAndLabel = features.join(phenotypeLabel).map({ case (patientID, (feature, realClass)) => (kmm.predict(transform(feature)), realClass)})


    val kMeansPurity = Metrics.purity(kmmClusterAssignmentAndLabel)
    println("KMM confusionMatrix: ")
    val KmeansConfusion = Metrics.confusionMatrix(kmmClusterAssignmentAndLabel)

    /** Kmeans, test k=2,4,5*/
    val kmm2=KMeans.train(featureVectors, k=2, maxIterations=20,1,"k-means||",0L)
    val kmmClusterAssignmentAndLabel2 = features.join(phenotypeLabel).map({ case (patientID, (feature, realClass)) => (kmm2.predict(transform(feature)), realClass)})
    println("KMM, K=2, PURITY: "+Metrics.purity(kmmClusterAssignmentAndLabel2))

    val kmm4=KMeans.train(featureVectors, k=4, maxIterations=20,1,"k-means||",0L)
    val kmmClusterAssignmentAndLabel4 = features.join(phenotypeLabel).map({ case (patientID, (feature, realClass)) => (kmm4.predict(transform(feature)), realClass)})
    println("KMM, K=4, PURITY: "+Metrics.purity(kmmClusterAssignmentAndLabel4))

    val kmm5=KMeans.train(featureVectors, k=5, maxIterations=20,1,"k-means||",0L)
    val kmmClusterAssignmentAndLabel5 = features.join(phenotypeLabel).map({ case (patientID, (feature, realClass)) => (kmm5.predict(transform(feature)), realClass)})
    println("KMM, K=5, PURITY: "+Metrics.purity(kmmClusterAssignmentAndLabel5))
    /** TODO: GMMM Clustering using spark mllib
      *  Train a Gaussian Mixture model using the variabe featureVectors as input
      *  Set maxIterations =20 and seed as 0L
      *  Assign each feature vector to a cluster(predicted Class)
      *  Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
      *  Find Purity using that RDD as an input to Metrics.purity
      *  Remove the placeholder below after your implementation
      **/
    import org.apache.spark.mllib.clustering.GaussianMixture
    import org.apache.spark.mllib.clustering.GaussianMixtureModel
    import org.apache.spark.mllib.linalg.Vectors
    val gmm = new GaussianMixture().setK(3).setMaxIterations(20).setSeed(0L).run(featureVectors)
    val predGMM = gmm.predict(featureVectors)

    val assignmentGMM = features.map({case (patientId,f)=>patientId}).zip(predGMM)

    val gmmClusterAssignmentAndLabel = assignmentGMM.join(phenotypeLabel).map({case (patientID,value)=>value})
    val gaussianMixturePurity = Metrics.purity(gmmClusterAssignmentAndLabel)
    println("GMM confusion matrix: ")
    val gmmConfusion = Metrics.confusionMatrix(gmmClusterAssignmentAndLabel)

    /** GMM k=2,4,5*/
    val gmm2 = new GaussianMixture().setK(2).setMaxIterations(20).setSeed(0L).run(featureVectors)
    val predGMM2 = gmm2.predict(featureVectors)
    val assignmentGMM2 = features.map({case (patientId,f)=>patientId}).zip(predGMM2)
    val gmmClusterAssignmentAndLabel2 = assignmentGMM2.join(phenotypeLabel).map({case (patientID,value)=>value})
    println("GMM, K=2,PURITY: "+Metrics.purity(gmmClusterAssignmentAndLabel2))

    val gmm4 = new GaussianMixture().setK(4).setMaxIterations(20).setSeed(0L).run(featureVectors)
    val predGMM4 = gmm4.predict(featureVectors)
    val assignmentGMM4 = features.map({case (patientId,f)=>patientId}).zip(predGMM4)
    val gmmClusterAssignmentAndLabel4 = assignmentGMM4.join(phenotypeLabel).map({case (patientID,value)=>value})
    println("GMM, K=4,PURITY: "+Metrics.purity(gmmClusterAssignmentAndLabel4))

    val gmm5 = new GaussianMixture().setK(5).setMaxIterations(20).setSeed(0L).run(featureVectors)
    val predGMM5 = gmm5.predict(featureVectors)
    val assignmentGMM5 = features.map({case (patientId,f)=>patientId}).zip(predGMM5)
    val gmmClusterAssignmentAndLabel5 = assignmentGMM5.join(phenotypeLabel).map({case (patientID,value)=>value})
    println("GMM, K=5,PURITY: "+Metrics.purity(gmmClusterAssignmentAndLabel5))
    /** NMF */
    val rawFeaturesNonnegative = rawFeatures.map({ case (patientID, f)=> Vectors.dense(f.toArray.map(v=>Math.abs(v)))})
    val (w, _) = NMF.run(new RowMatrix(rawFeaturesNonnegative), 3, 100)
    // for each row (patient) in W matrix, the index with the max value should be assigned as its cluster type
    val assignments = w.rows.map(_.toArray.zipWithIndex.maxBy(_._1)._2)
    // zip patientIDs with their corresponding cluster assignments
    // Note that map doesn't change the order of rows
    val assignmentsWithPatientIds=features.map({case (patientId,f)=>patientId}).zip(assignments)
    // join your cluster assignments and phenotypeLabel on the patientID and obtain a RDD[(Int,Int)]
    // which is a RDD of (clusterNumber, phenotypeLabel) pairs
    val nmfClusterAssignmentAndLabel = assignmentsWithPatientIds.join(phenotypeLabel).map({case (patientID,value)=>value})
    // Obtain purity value
    val nmfPurity = Metrics.purity(nmfClusterAssignmentAndLabel)
    println("nmf confusion matrix: ")
    val nmfConfusion = Metrics.confusionMatrix(nmfClusterAssignmentAndLabel)

    val (w2, _) = NMF.run(new RowMatrix(rawFeaturesNonnegative), 2, 100)
    val assignments2 = w2.rows.map(_.toArray.zipWithIndex.maxBy(_._1)._2)
    val assignmentsWithPatientIds2=features.map({case (patientId,f)=>patientId}).zip(assignments2)
    val nmfClusterAssignmentAndLabel2 = assignmentsWithPatientIds2.join(phenotypeLabel).map({case (patientID,value)=>value})
    println("nmf, k=2, purity: "+Metrics.purity(nmfClusterAssignmentAndLabel2))

    val (w4, _) = NMF.run(new RowMatrix(rawFeaturesNonnegative), 4, 100)
    val assignments4 = w4.rows.map(_.toArray.zipWithIndex.maxBy(_._1)._2)
    val assignmentsWithPatientIds4=features.map({case (patientId,f)=>patientId}).zip(assignments4)
    val nmfClusterAssignmentAndLabel4 = assignmentsWithPatientIds2.join(phenotypeLabel).map({case (patientID,value)=>value})
    println("nmf, k=4, purity: "+Metrics.purity(nmfClusterAssignmentAndLabel4))

    val (w5, _) = NMF.run(new RowMatrix(rawFeaturesNonnegative), 5, 100)
    val assignments5 = w5.rows.map(_.toArray.zipWithIndex.maxBy(_._1)._2)
    val assignmentsWithPatientIds5=features.map({case (patientId,f)=>patientId}).zip(assignments5)
    val nmfClusterAssignmentAndLabel5 = assignmentsWithPatientIds5.join(phenotypeLabel).map({case (patientID,value)=>value})
    println("nmf, k=5, purity: "+Metrics.purity(nmfClusterAssignmentAndLabel5))
    (kMeansPurity, gaussianMixturePurity, nmfPurity)
  }



  /**
   * load the sets of string for filtering of medication
   * lab result and diagnostics
    *
    * @return
   */
  def loadLocalRawData: (Set[String], Set[String], Set[String]) = {
    val candidateMedication = Source.fromFile("data/med_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateLab = Source.fromFile("data/lab_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateDiagnostic = Source.fromFile("data/icd9_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    (candidateMedication, candidateLab, candidateDiagnostic)
  }

  def loadRddRawData(sqlContext: SQLContext): (RDD[Medication], RDD[LabResult], RDD[Diagnostic]) = {
    /** You may need to use this date format. */
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")

    /** load data using Spark SQL into three RDDs and return them
      * Hint: You can utilize: edu.gatech.cse8803.ioutils.CSVUtils and SQLContext
      *       Refer to model/models.scala for the shape of Medication, LabResult, Diagnostic data type
      *       Be careful when you deal with String and numbers in String type
      * */

    /** TODO: implement your own code here and remove existing placeholder code below */
    val rawPatient = CSVUtils.loadCSVAsTable(sqlContext,"data/encounter_INPUT.csv","encounter")
    val rawMED = CSVUtils.loadCSVAsTable(sqlContext,"data/medication_orders_INPUT.csv","med")
    val MED = sqlContext.sql("select Member_ID as patientID,Order_Date as date,Drug_Name as medicine from med ")
    val rawLAB = CSVUtils.loadCSVAsTable(sqlContext,"data/lab_results_INPUT.csv","lab")
    val LAB = sqlContext.sql("select Member_ID as patientID, Date_Resulted as date, Result_Name as testName, Numeric_Result as value from lab where Numeric_Result <>'' ")
    val rawDIAG = CSVUtils.loadCSVAsTable(sqlContext,"data/encounter_dx_INPUT.csv","diag")
    val DIAG = sqlContext.sql("select e.Member_ID as patientID,e.Encounter_DateTime as date,d.code as code from encounter e join diag d on e.Encounter_ID = d.Encounter_ID")

    val medication: RDD[Medication] =MED.map(x => Medication(x.getString(0).toLowerCase, dateFormat.parse(x.getString(1)),x.getString(2).toLowerCase))
    val labResult: RDD[LabResult] =  LAB.map(x=>LabResult(x.getString(0).toLowerCase,dateFormat.parse(x.getString(1)),x.getString(2).toLowerCase,x.getString(3).replaceAll(",","").toDouble))
    val diagnostic: RDD[Diagnostic] =  DIAG.map(x=>Diagnostic(x.getString(0).toLowerCase,dateFormat.parse(x.getString(1)),x.getString(2).toLowerCase))

    (medication, labResult, diagnostic)
  }

  def createContext(appName: String, masterUrl: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl)
    new SparkContext(conf)
  }

  def createContext(appName: String): SparkContext = createContext(appName, "local")

  def createContext: SparkContext = createContext("CSE 8803 Homework Two Application", "local")
}
