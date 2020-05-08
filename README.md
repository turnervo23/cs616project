# Using Machine Learning to Predict Bug Severity

Hi anyone who's reading this, it's Turner. This is the artifact for my CS 616 project.

To get started, download the bug prediction dataset from http://bug.inf.usi.ch/download.php (click "download everything" on all). In order for the preprocessing function to work it's important that everything's formatted correctly. Extract the files and put everything in a folder called "data" so the directory hierarchy looks something like this:

data<br/>
----eclipse<br/>
--------biweekly-ck-values<br/>
------------ ...<br/>
--------biweekly-oo-values<br/>
------------ ...<br/>
--------churn<br/>
------------ ...<br/>
--------entropy<br/>
------------ ...<br/>
--------bug-metrics.csv<br/>
--------change-metrics.csv<br/>
--------complexity-code-change.csv<br/>
--------single-version-ck-oo.csv<br/>
----equinox<br/>
-------- ...<br/>
----lucene<br/>
-------- ...<br/>
----mylyn<br/>
-------- ...<br/>
----pde<br/>
-------- ...

The csvs are semicolon delimited, I changed them to comma delimited because I couldn't figure out how to open them in Excel. I did this by just changing every semicolon to a comma. On Linux you can do this with

find . -type f -exec sed -i 's/;/,/g' {} +

and it will change every semicolon to a comma in all files in a directory recursively. On Windows or Mac I'm sure there's a similar way.

Lastly, make sure the code is in a separate directory at the same level as "data":

data<br/>
---- ...<br/>
code<br/>
----BPD_classification.py<br/>
----BPD_feature_importance.py<br/>
----BPD_numbugsvsseverity.py<br/>
----BPD_preprocess.py

To use, run one of the classification scripts (BPD_classification.py or BPD_feature_importance.py). You can comment/uncomment different elements of the lists and write different for loops in order to run the different experiments. Or you can come up with your own. Have fun!
