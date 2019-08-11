**REQUIREMENTS**    
numpy==1.15.1      
wfdb==2.2.1      
<br>

**GENERAL INFO**   
Execute `af_classifier.py` to generate the results and save them through pickle.           
Execute `results.py` to save the classification for every record in `afdb_result.csv` and print the metrics.
       
`dataset/afdb_result.csv` contains the TP, TN, FP, FN for all the records         
`dataset/afdb_result` is a an array of [record_name, tp, tn, fp, fn] that can be read through pickle

**CURRENT RESULT ON AFDB**     
<pre>
TRUE POSITIVE :  498963        
TRUE NEGATIVE :  684209         
FALSE POSITIVE:  17552       
FALSE NEGATIVE:  20835     
</pre>

-----------------------------------

<pre>
SENSITIVITY   :  0.9599171216511029       
SPECIFICITY   :  0.9749886357321083      
PRECISION     :  0.9660184118563836        
ACCURACY      :  0.9685754024160929        
</pre>