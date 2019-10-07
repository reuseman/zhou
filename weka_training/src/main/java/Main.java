import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.TimeUnit;


public class Main {

    public static class Stats {
        public double pctCorrect;
        public double pctIncorrect;
        public double pctUnclassified;

        public Stats(Double pctCorrect, Double pctIncorrect, Double pctUnclassified) {
            this.pctCorrect = pctCorrect;
            this.pctIncorrect = pctIncorrect;
            this.pctUnclassified = pctUnclassified;
        }
    }

    public static void saveConfusionMatrixToCsv(double[][] matrix, Path path) throws IOException {
        int tn = (int) matrix[0][0];
        int fp = (int) matrix[0][1];
        int fn = (int) matrix[1][0];
        int tp = (int) matrix[1][1];

        String content = String.format("tp,fp,fn,tn\n%d,%d,%d,%d", tp, fp, fn, tn);
        BufferedWriter writer = new BufferedWriter(new FileWriter(path.toFile()));

        writer.write(content);
        writer.close();
    }

    public static Stats execute(Classifier classifier, Instances train, Instances test, Path csvPath) throws Exception {
        classifier.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(classifier, test);
        saveConfusionMatrixToCsv(eval.confusionMatrix(), csvPath);

        return new Stats(eval.pctCorrect(), eval.pctIncorrect(), eval.pctUnclassified());
    }

    public static void main(String[] args) {
        // Get path where the main folder is located
        // tp, tn, fp, fn
        String homePath = System.getProperty("user.home");
        Path datasetPath = Paths.get(homePath, "Uni", "Tirocinio", "code", "zhou", "dataset", "generated", "explicit_entropy_fft_16_ar_4");
        File mainFolder = new File(datasetPath.toUri());

        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("HH:mm:ss dd/MM/yyyy");
        int folderNumber = 0;
        int totalFolders = mainFolder.listFiles().length;

        // Added just for Fourier dataset
        Normalize normalize = new Normalize();
        Discretize discretize = new Discretize();
        discretize.setBinRangePrecision(20);

        for (final File currentFolder : mainFolder.listFiles()) {
            File[] datasets = currentFolder.listFiles();
            Path resultsCsv = null;
            folderNumber++;

            try {
                Instances testDataset = null;
                Instances trainDataset = null;
                Stats stats = null;

                // Used to keep track of the previously evaluated models
                boolean j48, ibk, logistic, jrip, bayesnet, adaboostm1, randomforest, reptree;
                j48 = ibk = logistic = jrip = bayesnet = adaboostm1 = randomforest = reptree = false;

                long startTime = 0l;
                long duration = 0l;
                long minutes = 0l;

                for (File f : datasets) {
                    switch (f.getName()) {
                        case "train.arff":
                            DataSource source = new DataSource(f.getAbsolutePath());
                            trainDataset = source.getDataSet();
                            trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
                            // trainDataset = Filter.useFilter(trainDataset, normalize);
                            break;
                        case "test.arff":
                            source = new DataSource(f.getAbsolutePath());
                            testDataset = source.getDataSet();
                            testDataset.setClassIndex(testDataset.numAttributes() - 1);
                            // testDataset = Filter.useFilter(testDataset, normalize);
                            break;
                        case "bayesnet.csv":
                            bayesnet = true;
                            break;
                        case "logistic.csv":
                            logistic = true;
                            break;
                        case "ibk.csv":
                            ibk = true;
                            break;
                        case "adaboostm1.csv":
                            adaboostm1 = true;
                            break;
                        case "jrip.csv":
                            jrip = true;
                            break;
                        case "randomforest.csv":
                            randomforest = true;
                            break;
                        case "j48.csv":
                            j48 = true;
                            break;
                        case "reptree.csv":
                            reptree = true;
                            break;
                    }
                }

                System.out.println("=======================================");
                System.out.println("  CURRENT FOLDER: " + currentFolder.getName() +
                        " | (" + folderNumber + "/" + totalFolders + ")");
                System.out.println("         started: " + dtf.format(LocalDateTime.now()));
                System.out.println("=======================================");

                if (logistic) {
                    System.out.println("Logistic              has been already evaluated previously!");
                } else {
                    startTime = System.nanoTime();

                    resultsCsv = Paths.get(currentFolder.getAbsolutePath(), "logistic.csv");
                    stats = execute(new Logistic(), trainDataset, testDataset, resultsCsv);

                    duration = System.nanoTime() - startTime;
                    minutes = TimeUnit.MINUTES.convert(duration, TimeUnit.NANOSECONDS);
                    System.out.println("Logistic              evaluated in " + minutes + "minutes, with: " +
                            stats.pctCorrect + " correct,  " + stats.pctIncorrect + " incorrect,  " +
                            stats.pctUnclassified + " unclassified.");
                }

                /*System.gc();

                if (ibk) {
                    System.out.println("IBk                   has been already evaluated previously!");
                } else {
                    startTime = System.nanoTime();

                    resultsCsv = Paths.get(currentFolder.getAbsolutePath(), "ibk.csv");
                    stats = execute(new IBk(), trainDataset, testDataset, resultsCsv);

                    duration = System.nanoTime() - startTime;
                    minutes = TimeUnit.MINUTES.convert(duration, TimeUnit.NANOSECONDS);
                    System.out.println("IBk                   evaluated in " + minutes + "minutes, with: " +
                            stats.pctCorrect + " correct,  " + stats.pctIncorrect + " incorrect,  " +
                            stats.pctUnclassified + " unclassified.");
                }*/

                System.gc();

                if (adaboostm1) {
                    System.out.println("AdaBoostM1            has been already evaluated previously!");
                } else {
                    startTime = System.nanoTime();

                    resultsCsv = Paths.get(currentFolder.getAbsolutePath(), "adaboostm1.csv");
                    stats = execute(new AdaBoostM1(), trainDataset, testDataset, resultsCsv);

                    duration = System.nanoTime() - startTime;
                    minutes = TimeUnit.MINUTES.convert(duration, TimeUnit.NANOSECONDS);
                    System.out.println("AdaBoostM1            evaluated in " + minutes + "minutes, with: " +
                            stats.pctCorrect + " correct,  " + stats.pctIncorrect + " incorrect,  " +
                            stats.pctUnclassified + " unclassified.");
                }

                /*
                System.gc();

                if (jrip) {
                    System.out.println("JRip                  has been already evaluated previously!");
                } else {
                    startTime = System.nanoTime();

                    resultsCsv = Paths.get(currentFolder.getAbsolutePath(), "jrip.csv");
                    stats = execute(new JRip(), trainDataset, testDataset, resultsCsv);

                    duration = System.nanoTime() - startTime;
                    minutes = TimeUnit.MINUTES.convert(duration, TimeUnit.NANOSECONDS);
                    System.out.println("JRip                  evaluated in " + minutes + "minutes, with: " +
                            stats.pctCorrect + " correct,  " + stats.pctIncorrect + " incorrect,  " +
                            stats.pctUnclassified + " unclassified.");
                }*/

                System.gc();

                if (randomforest) {
                    System.out.println("RandomForest          has been already evaluated previously!");
                } else {
                    startTime = System.nanoTime();

                    resultsCsv = Paths.get(currentFolder.getAbsolutePath(), "randomforest.csv");
                    stats = execute(new RandomForest(), trainDataset, testDataset, resultsCsv);

                    duration = System.nanoTime() - startTime;
                    minutes = TimeUnit.MINUTES.convert(duration, TimeUnit.NANOSECONDS);
                    System.out.println("RandomForest          evaluated in " + minutes + "minutes, with: " +
                            stats.pctCorrect + " correct,  " + stats.pctIncorrect + " incorrect,  " +
                            stats.pctUnclassified + " unclassified.");
                }

                System.gc();

                if (j48) {
                    System.out.println("J48                   has been already evaluated previously!");
                } else {
                    startTime = System.nanoTime();

                    resultsCsv = Paths.get(currentFolder.getAbsolutePath(), "j48.csv");
                    stats = execute(new J48(), trainDataset, testDataset, resultsCsv);

                    duration = System.nanoTime() - startTime;
                    minutes = TimeUnit.MINUTES.convert(duration, TimeUnit.NANOSECONDS);
                    System.out.println("J48                   evaluated in " + minutes + "minutes, with: " +
                            stats.pctCorrect + " correct,  " + stats.pctIncorrect + " incorrect,  " +
                            stats.pctUnclassified + " unclassified.");
                }

                System.gc();

                if (reptree) {
                    System.out.println("REPTree               has been already evaluated previously!");
                } else {
                    startTime = System.nanoTime();

                    resultsCsv = Paths.get(currentFolder.getAbsolutePath(), "reptree.csv");
                    stats = execute(new REPTree(), trainDataset, testDataset, resultsCsv);

                    duration = System.nanoTime() - startTime;
                    minutes = TimeUnit.MINUTES.convert(duration, TimeUnit.NANOSECONDS);
                    System.out.println("REPTree               evaluated in " + minutes + "minutes, with: " +
                            stats.pctCorrect + " correct,  " + stats.pctIncorrect + " incorrect,  " +
                            stats.pctUnclassified + " unclassified.\n\n");
                }

                System.gc();

                if (bayesnet) {
                    System.out.println("BayesNet              has been already evaluated previously!");
                } else {
                    startTime = System.nanoTime();
                    // Added just for Fourier dataset
                    trainDataset = Filter.useFilter(trainDataset, discretize);
                    testDataset = Filter.useFilter(testDataset, discretize);

                    resultsCsv = Paths.get(currentFolder.getAbsolutePath(), "bayesnet.csv");
                    stats = execute(new BayesNet(), trainDataset, testDataset, resultsCsv);

                    duration = System.nanoTime() - startTime;
                    minutes = TimeUnit.MINUTES.convert(duration, TimeUnit.NANOSECONDS);
                    System.out.println("BayesNet              evaluated in " + minutes + "minutes, with: " +
                            stats.pctCorrect + " correct,  " + stats.pctIncorrect + " incorrect,  " +
                            stats.pctUnclassified + " unclassified.");
                }

                System.gc();

            } catch (IOException e) {
                System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                System.out.println("ERROR while saving to " + resultsCsv);
                System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                e.printStackTrace();
            } catch (Exception e) {
                System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                e.printStackTrace();
                System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
            }
        }

    }
}
