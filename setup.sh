pip install --user nltk
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('stopwords')"

pip install -U pip setuptools wheel
pip install -r requirements.txt

pip install --upgrade gensim
pip install -U spacy
pip install -U scikit-learn
python -m spacy download en_core_web_sm