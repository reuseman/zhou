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

**A) CURRENT RESULT WITH (same results with .qrsc):**     
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

with `05091.qrsc`, `07859.qrsc`
<pre>
TRUE POSITIVE :  500519
TRUE NEGATIVE :  684414
FALSE POSITIVE:  17563
FALSE NEGATIVE:  20599
</pre>

<pre>
SENSITIVITY   :  0.960471524683469
SPECIFICITY   :  0.9749806617595733
PRECISION     :  0.9660999610100331
ACCURACY      :  0.9687988259293023
</pre>

-----------------------------------

**B) CURRENT RESULT WITH:**     
1 if percentage > 50%, else 0
<pre>
TRUE POSITIVE :  498965
TRUE NEGATIVE :  684148
FALSE POSITIVE:  17550
FALSE NEGATIVE:  20821
</pre>

<pre>
SENSITIVITY   :  0.9599431304421435
SPECIFICITY   :  0.974989240385465
PRECISION     :  0.9660222839607756
ACCURACY      :  0.9685865717438787
</pre>

-----------------------------------
with `05091.qrsc`, `07859.qrsc`
<pre>
TRUE POSITIVE :  500588
TRUE NEGATIVE :  684188
FALSE POSITIVE:  17494
FALSE NEGATIVE:  20825
</pre>

<pre>
SENSITIVITY   :  0.9600604511203211
SPECIFICITY   :  0.975068478313538
PRECISION     :  0.9662331445601275
ACCURACY      :  0.9686704630466153
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

-----------------------------------
with `05091.qrsc`, `07859.qrsc`
<pre>
TRUE POSITIVE :  500519
TRUE NEGATIVE :  684114
FALSE POSITIVE:  17279
FALSE NEGATIVE:  20599
</pre>

<pre>
SENSITIVITY   :  0.960471524683469
SPECIFICITY   :  0.9753647384561864
PRECISION     :  0.9666298440704676
ACCURACY      :  0.9690162297108165
</pre>

-----------------------------------

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