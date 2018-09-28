from luigi.contrib.external_program import ExternalProgramTask
from luigi.contrib.spark import PySparkTask
from luigi.parameter import IntParameter, Parameter
from luigi import LocalTarget, Task
import luigi

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
    
class DownloadDataset(ExternalProgramTask):

    dataset_name = Parameter(default="reddit_ds_got")

    base_url = "http://plainpixels.work/resources/datasets"
    file_fomat = "zip"

    def output(self):
        return LocalTarget("../dataset/%s.%s" % (self.dataset_name, self.file_fomat))

    def program_args(self):
        url = "%s/%s.%s" % (self.base_url, 
                            self.dataset_name, 
                            self.file_fomat)
        return ["curl", "-L",
                "-o", self.output().path,
                url]
    
    
class ExtractDataset(ExternalProgramTask):
    
    dataset_name = Parameter(default="reddit_ds_got")
    
    def requires(self):
        return DownloadDataset(self.dataset_name)

    def output(self):
        return LocalTarget("datasets/%s/raw" % self.dataset_name)

    def program_args(self):
        self.output().makedirs()
        return ["unzip", "-u", "-q",
                "-d", self.output().path,
                self.input().path]
    

class Clean(Task):
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    dataset_name = Parameter(default="reddit_ds_got")
    training_data = "training/data.csv"
    
    # Der verwendete Tokenizer
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    
    # Die Liste von Stop-Woertern
    # die herausgefiltert werden
    stopwords = nltk.corpus.stopwords.words('english')
    
     # Der Stemmer fuer Englische Woerter
    stemmer = nltk.SnowballStemmer("english")
    
    # Als Abhaengigkeit wird der
    # Task *Download* zurueckgegeben
    def requires(self):
        return ExtractDataset(self.dataset_name)
    
    # Das LocalTarget fuer die sauberen Daten
    # Die Daten werden unter
    # "model/<version>/cleaned.csv gespeichert
    def output(self):
        return LocalTarget("datasets/%s/cleaned/cleaned.csv" % self.dataset_name)

    # Die Rohdaten werden zerstueckelt
    # durch die Stopwort-Liste gefiltert
    # und auf ihre Wortstaemme zurueckgefuehrt
    def run(self):
        import pandas
        data = "%s/%s" % (self.input().path, self.training_data)
        dataset = pandas.read_csv(data, encoding='utf-8', sep=';').fillna('')
        dataset["cleaned_words"] = dataset.apply(self.clean_words, axis=1)
        with self.output().open("w") as out:
            dataset[["cleaned_words", "subreddit"]].to_csv(out,  encoding='utf-8', index=False, sep=';')

    def clean_words(self, post):
        tokenized = self.tokenizer.tokenize(post["title"] + " " + post["selftext"])
        lowercase = [word.lower() for word in tokenized]
        filtered = [word for word in lowercase if word not in self.stopwords]
        stemmed = [self.stemmer.stem(word) for word in filtered]
        return " ".join(stemmed)
    
    
class TrainModel(PySparkTask):
    dataset_name = Parameter(default="reddit_ds_got")
    version = IntParameter(default=1)
    
    # PySpark Parameter
    driver_memory = '1g'
    executor_memory = '2g'
    executor_cores = '2'
    num_executors = '4'
    master = 'local'
    
    # Als Abhaengigkeit wird der
    # Task *Clean* zurueckgegeben
    def requires(self):
        return Clean(self.dataset_name)
    
    # Das LocalTarget fuer das Model
    # Die Daten werden unter
    # "model/<version>/model gespeichert
    def output(self):
        return LocalTarget("model/%d" % self.version)

    def main(self, sc, *args):
        from pyspark.sql.session import SparkSession
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import HashingTF, Tokenizer
        from pyspark.ml.classification import DecisionTreeClassifier
        
        # Initialisiere den SQLContext
        sql = SparkSession.builder\
            .enableHiveSupport() \
            .config("hive.exec.dynamic.partition", "true") \
            .config("hive.exec.dynamic.partition.mode", "nonstrict") \
            .config("hive.exec.max.dynamic.partitions", "4096") \
            .getOrCreate()
        
        # Lade die bereinigten Daten
        df = sql.read.format("com.databricks.spark.csv") \
            .option("header", "true") \
            .option("delimiter", ";") \
            .load(self.input().path)
        
        # Den Klassifikator trainieren
        labeled = df.withColumn("label", df.subreddit.like("datascience").cast("double"))
        train_set, test_set = labeled.randomSplit([0.8, 0.2])
        tokenizer = Tokenizer().setInputCol("cleaned_words").setOutputCol("tokenized")
        hashing = HashingTF().setNumFeatures(1000).setInputCol("tokenized").setOutputCol("features")
        decision_tree = DecisionTreeClassifier()
        pipeline = Pipeline(stages=[tokenizer, hashing, decision_tree])
        model = pipeline.fit(train_set)
        model.save(self.output().path)