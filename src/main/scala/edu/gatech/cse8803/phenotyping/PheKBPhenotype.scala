/**
  * @author Ting Pan <tpan35@gatech.edu>,
  */

package edu.gatech.cse8803.phenotyping

import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import org.apache.spark.rdd.RDD

object T2dmPhenotype {
  /**
    * Transform given data set to a RDD of patients and corresponding phenotype
    * @param medication medication RDD
    * @param labResult lab result RDD
    * @param diagnostic diagnostic code RDD
    * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
    */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {
    

    val sc = medication.sparkContext
    val TotalPatients = diagnostic.map(_.patientID).distinct()
    /** Sanity check
    *diagnostic.map(_.patientID).distinct().count()
    *labResult.map(_.patientID).distinct().count()*/

    /** Hard code the criteria */
    val type1_dm_dx = Set("250.03","250.01","250.11","250.13","250.21","250.23","250.31","250.33","250.41","250.43","250.51","250.53","250.61","250.63","250.71","250.73","250.81","250.83","250.91","250.93")
    val type1_dm_med = Set("med1", "insulin nph","lantus","insulin glargine","insulin aspart","insulin detemir","insulin lente","insulin reg","insulin,ultralente")
    val type2_dm_dx = Set("250.3","250.32","250.2","250.22","250.9","250.92","250.8","250.82","250.7","250.72","250.6","250.62","250.5","250.52","250.4","250.42","250.00","250.02")
    val type2_dm_med = Set("chlorpropamide","diabinese","diabanase","diabinase","glipizide","glucotrol","glucotrol xl","glucatrol ","glyburide","micronase","glynase","diabetamide","diabeta","glimepiride","amaryl","repaglinide","prandin","nateglinide","metformin","rosiglitazone","pioglitazone","acarbose","miglitol","sitagliptin","exenatide","tolazamide","acetohexamide","troglitazone","tolbutamide","avandia","actos","actos","glipizide")
    /** Find CASE Patients */

      /** Ntype1DM:3002*/

    val type1DM = diagnostic.filter(d => type1_dm_dx.contains(d.code)).map(_.patientID).distinct()
    val Ntype1DM = diagnostic.map(_.patientID).distinct().subtract(type1DM)
    /** Ntype1DM AND type2DM : 1265*/
    val type2DM = diagnostic.filter(d => type2_dm_dx.contains(d.code)).map(_.patientID).distinct()
    val Ntype1DM_type2DM = Ntype1DM.intersection(type2DM)

     /** Ntype1DM AND type2DM AND Ntype1_med: 427, Ntype1DM AND type2DM AND type1_med: 838 */
    val type1DMmed = medication.filter(m => type1_dm_med.contains(m.medicine)).map(_.patientID).distinct()

    val NYtype1DMmed = Ntype1DM_type2DM.intersection(type1DMmed)
    val NYNtype1DMmed = Ntype1DM_type2DM.subtract(NYtype1DMmed)

   /** NYtype1DMmed AND Ntype2DMmed: 255, NYtype1DMmed AND Ytype2DMmed: 583*/
     val type2DMmed = medication.filter(m => type2_dm_med.contains(m.medicine)).map(_.patientID).distinct()
    val NYYtype2DMmed = NYtype1DMmed.intersection(type2DMmed)
    val NYYNtype2DMmed = NYtype1DMmed.subtract(NYYtype2DMmed)

    /** type2 medicaiton proceeds type 1: 294
      * check which type of medication a patient get first in his/her life.
      * Get the first occurrence of Type1 medication and the first occurrence of Type2 medication for a given patient and compare them.
      * reduceByKey
      * getTime()*/
      val patients = NYYtype2DMmed.collect()
      val bothMeds = medication.filter(m => patients.contains(m.patientID)).filter(m=> type1_dm_med.contains(m.medicine) || type2_dm_med.contains(m.medicine))

    val groupMed = bothMeds.sortBy(_.date).groupBy(_.patientID)

    val FirstDate = groupMed.map(a=>a._2.head)
    val proceeds = FirstDate.filter(f=> !type1_dm_med.contains(f.medicine)).map(_.patientID).distinct()
    val caseP = sc.union(NYNtype1DMmed,NYYNtype2DMmed, proceeds)
    val casePatients = caseP.map(x=>(x,1)).cache()


    /** Find CONTROL Patients */

    val glucose = labResult.filter{a => a.testName.equals("glucose")  && a.value<=110.0 }
    val serum = labResult.filter{a => a.testName.contains("serum") &&a.testName.contains("glucose") && a.value<=110.0}
    val glucose_fasting = labResult.filter{a=> a.testName.contains("fasting") && a.testName.contains("glucose") && a.value<110.0}
    val HbA1C = labResult.filter{a => a.testName.contains("hba1c") && a.value<6.0}
    val A1C = labResult.filter{a => a.testName.contains("a1c") && a.value<6.0}
    val normal_glu = sc.union(glucose, serum, glucose_fasting, HbA1C, A1C).map(_.patientID).distinct()


    val dm_related = Set("790.21","790.22", "790.2", "790.29", "648.81", "648.82", "648.83", "648.84", "648.0", "648.00", "648.01", "648.02", "648.03", "648.04", "791.5", "277.7", "V77.1", "256.4", "250.7", "250.72", "250.6", "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.01", "250.02")
    val NDM_related = diagnostic.filter(x=> dm_related.contains(x.code)).map(_.patientID).distinct()
    val control = normal_glu.subtract(NDM_related)
    val controlPatients = control.map(x=>(x,2)).cache()



    /** Find OTHER Patients */
      val otherP = TotalPatients.subtract(caseP).subtract(control)
    val others = otherP.map(x=>(x,3)).cache()


    /** Once you find patients for each group, make them as a single RDD[(String, Int)] */

    val phenotypeLabel = sc.union(casePatients, controlPatients, others)
    /**phenotypeLabel.coalesce(1,true).saveAsTextFile("test/phenotypeLabel")*/
    phenotypeLabel.map{case (patientID, phenotype) => (phenotype, 1.0)}.reduceByKey(_+_).collect()

      /** Return */
    phenotypeLabel
  }
}
