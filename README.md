# Sarcasm Detection
This is a university project to classify sarcasm on the [SARC reddit dataset](https://paperswithcode.com/dataset/sarc). 
Sarcasm is very difficult to classify as it is very dependent on the context of the speaker as well as text. 
We implement a BERT, SVM, Random Forest and CNN model, each in a respective jupyter notebook.
Our best model is an augmented CNN that achieves a mean accuracy of 73.1% over 5 folds (see [report](docs/report.pdf)).

## Getting Started 
1. Install Python 3.9
2. Install dependencies with `pip install -r requirements.txt`
3. Run a respective jupyter notebook (`*.ipynb`)
