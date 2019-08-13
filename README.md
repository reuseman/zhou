**REQUIREMENTS**    
numpy==1.15.1      
wfdb==2.2.1      
<br>

-----------

**GENERAL INFO**   
Execute `af_classifier.py` to generate the results and save them through pickle.           
Execute `results.py` to save the classification for every record in `afdb_result.csv` and print the metrics.
       
`dataset/afdb_result.csv` contains the TP, TN, FP, FN for all the records         
`dataset/afdb_result` is a an array of [record_name, tp, tn, fp, fn] that can be read through pickle

-----------

**A) CURRENT RESULT WITH:**     
1 if percentage = 100%, else 0
<pre>
TRUE POSITIVE :  498896
TRUE NEGATIVE :  684374
FALSE POSITIVE:  17619
FALSE NEGATIVE:  20595
</pre>

<pre>
SENSITIVITY   :  0.960355424829304
SPECIFICITY   :  0.9749014591313588
PRECISION     :  0.9658886963592539
ACCURACY      :  0.9687151039227694
</pre>

-----------------------------------

**B) CURRENT RESULT WITH:**     
1 if percentage > 50%, else 0
<pre>
TRUE POSITIVE :  499071
TRUE NEGATIVE :  684109
FALSE POSITIVE:  17444
FALSE NEGATIVE:  20860
</pre>

<pre>
SENSITIVITY   :  0.9598792916752416
SPECIFICITY   :  0.9751351644138077
PRECISION     :  0.9662275054935481
ACCURACY      :  0.9686414230558894
</pre>

-----------------------------------

**C) CURRENT RESULT WITH:**     
Hybrid not classified: 584
<pre>
TRUE POSITIVE :  498896
TRUE NEGATIVE :  684074
FALSE POSITIVE:  17335
FALSE NEGATIVE:  20595
</pre>

<pre>
SENSITIVITY   :  0.960355424829304
SPECIFICITY   :  0.9752854611218277
PRECISION     :  0.9664200716345976
ACCURACY      :  0.9689327545253501
</pre>

---------------------------

### ALGORITHM:
<pre>
AF BEATS:      516515
NON AF BEATS:  704969
</pre>

### PAPER:
<pre>
AF BEATS:      519687
NON AF BEATS:  701887
</pre>