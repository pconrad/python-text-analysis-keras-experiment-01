# python-text-analysis-keras-experiment-01
An experiment with python text analysis using keras.   A primer on Machine Learning also


The contents of this repo are based on working with this tutorial:

* https://realpython.com/python-keras-text-classification/


# Tagged Data

The tagged data in zip file: sentiment_labelled_sentences.zip (and in the directory sentiment_labelled_sentences) comes
from this link: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

The original paper from which this data comes is here:

Dimitrios Kotzias, Misha Denil, Nando de Freitas, and Padhraic Smyth. 2015. From Group to Individual Labels Using Deep Features. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '15). ACM, New York, NY, USA, 597-606. DOI: https://doi.org/10.1145/2783258.2783380

Each of the three files contains lines that have a sentence followed by a score of either 0 or 1.
The sentence is separated from the score by a tab character, which in Python is `\t`.

So this line of code reads a file in this format into dictionary.  While the file is not, strictly
speaking, a "Comma Separated Value" file (strictly speaking, its a "tab separated value" file,), we
can use the software for CSV files to read it:

```python
   df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
```

So that line of code reads one of these files into something called a Pandas Data Frame.

```
>>> import pandas as pd
>>> filepath='data/sentiment_analysis/imdb_labelled.txt'
>>> df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
>>> type(df)
<class 'pandas.core.frame.DataFrame'>
>>> 
```

The method `.head()` of a Pandas Data Frame will show me the first few lines of the data:

```
>>> df.head()
                                            sentence  label
0  A very, very, very slow-moving, aimless movie ...      0
1  Not sure who was more lost - the flat characte...      0
2  Attempting artiness with black & white and cle...      0
3       Very little music or anything to speak of.        0
4  The best scene in the movie was when Gerardo i...      1
>>> 
```


# Libraries Used

The code in this repo uses [Pandas](https://pandas.pydata.org/), 
which is a Python Data Analysis Library.

Before you start using the code, you will likely need to install Pandas.  Here's how:

```
pip install pandas
```

On some machines, e.g. those with both Python 2 and Python 3 installed, where you use
the command `python3` to bring up Python, you may need to use:

```
pip3 install pandas
```

# More about Pandas

You can read more about Pandas in this book, which is available with full-text access from UCSB IP addresses,
or from off campus using your UCSBNetID and the [Campus VPN](http://www.ets.ucsb.edu/services/campus-vpn/get-connected):

[Python for Data Analysis, 2nd Edition](https://proquest.safaribooksonline.com/book/programming/python/9781491957653)
