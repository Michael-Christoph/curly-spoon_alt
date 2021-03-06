import weka.core.converters.ArffSaver;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;

/**
 * Created by Michael on 02.09.2017.
 */
public class TestClusterParams {
    private static final boolean SAVE_INSTANCES_TO_ARFF_FILE = false;
    private static final String PATH_TRAINING = "corpus_jpg15vh+autokorr+s10/";
    private static final String PATH_TEST = "query_jpg15vh+autokorr+s10/";
    public static void main(String[] args){
        String trialName = "mp_sa1118";
        File dir = new File("logs_" + trialName);
        dir.mkdir();
        File resultFile = new File("results_" + trialName + ".csv");
        for (int cacheSize = 4;cacheSize<=16;cacheSize *=2){
            for (int numWords = 4; numWords<=16;numWords *=2){
                for (int i = 0; i<=1;i++){
                    boolean useCanopiesForFasterClustering = (i==0) ? false : true;
                    for (int maxCandidates=100;maxCandidates<=1000;maxCandidates *=10){
                        for (int periodicPruning=10000;periodicPruning>=1000;periodicPruning /=10){
                            for (int minDensity=2;minDensity<=4;minDensity *= 2){
                                for (int maxNumIterations=500;maxNumIterations>=100;maxNumIterations /=5){
                                    for (int j=0;j<=1;j++){
                                        boolean faster = (j==0) ? false : true;
                                        for (int numSlots=1;numSlots<=2;numSlots++){
                                            runOneConfiguration(cacheSize,
                                                    numWords,
                                                    useCanopiesForFasterClustering,
                                                    maxCandidates,
                                                    periodicPruning,
                                                    minDensity,
                                                    -1.0,
                                                    -1.25,
                                                    maxNumIterations,
                                                    faster,
                                                    numSlots,
                                                    false,
                                                    SAVE_INSTANCES_TO_ARFF_FILE,
                                                    resultFile,trialName);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        System.out.println("Finished test algorithm.");

    }
    private static void runOneConfiguration(int cacheSize,int numWords,
                                            boolean useCanopiesForFasterClustering,
                                            int maxCandidates, int periodicPruning,
                                            int minDensity,double t2,double t1,
                                            int maxNumIterations,boolean faster,
                                            int numSlots, boolean iWantDebugInfo,
                                            boolean saveInstancesToArff,
                                            File resultFile,String trialName){
        long startTime = System.currentTimeMillis();
        String useCanopies = (useCanopiesForFasterClustering) ? "t" : "f";
        String fasterToString = (faster) ? "t" : "f";

        BagOfWordsGenerator_mod myBgfwrdgen = new BagOfWordsGenerator_mod(cacheSize,
                numWords,
                useCanopiesForFasterClustering, maxCandidates,
                periodicPruning,minDensity,t2,t1,maxNumIterations,
                faster,numSlots,iWantDebugInfo);
        String[] tuerschilder = new String[]{"pt_3-0-13","pt_3-0-26","pt_3-0-56",
                "pt_3-0-57","pt_3-0-67","pt_3-0-68","pt_3-0-84a","pt_3-0-84b",
                "pt_3-0-84c","pt_3-0-84d"};

        try {
            //MP
            String corpusFormat = "jpg";
            if (PATH_TEST.indexOf('j') == -1)
                corpusFormat = "png";

            //MP-Versuch
            int trainingExamplesProTuerschild = 3;
            int wenigerTuerschilderDamitHeapNichtUeberlaeuft = 8;
            for (int sch = 0; sch<tuerschilder.length-wenigerTuerschilderDamitHeapNichtUeberlaeuft;sch++){
                for (int i=1; i<=trainingExamplesProTuerschild;i++){
                    myBgfwrdgen.addTrainingExample(PATH_TRAINING + tuerschilder[sch] + "/pic_" + Integer.toString(i) + "." + corpusFormat);
                }
            }

            String configName = PATH_TRAINING.substring(0,PATH_TRAINING.length()-1) +
                    PATH_TEST.substring(0,PATH_TEST.length()-1) +
                    cacheSize+numWords+useCanopies+
                    maxCandidates+periodicPruning+minDensity+t2+t1+
                    maxNumIterations+fasterToString+numSlots+trainingExamplesProTuerschild+
                    wenigerTuerschilderDamitHeapNichtUeberlaeuft;


            if (saveInstancesToArff){
                ArffSaver saver = new ArffSaver();
                saver.setInstances(myBgfwrdgen.getData());
                saver.setFile(new File(".data" + trialName + "/" + configName +".arff"));
                saver.writeBatch();
            }

            //t.addTrainingExample(pathTraining + "PT_3.0.61/20170602_134243.png");

            myBgfwrdgen.cluster();

            //MP
            String queryFormat = "jpg";
            if (PATH_TEST.indexOf('j') == -1)
                queryFormat = "png";

            int u[] = myBgfwrdgen.generateBoWForImage(PATH_TEST + "pt_3-0-13/pic_19." + queryFormat,
                    "logs_" + trialName + "/" + "log_u" + configName + ".csv",true);
            int v[] = myBgfwrdgen.generateBoWForImage(PATH_TEST + "pt_3-0-13/pic_20." + queryFormat,
                    "logs_" + trialName + "/" + "log_v" + configName + ".csv",true);
            int w[] = myBgfwrdgen.generateBoWForImage(PATH_TEST + "pt_3-0-26/pic_19." + queryFormat,
                    "logs_" + trialName + "/" + "log_w" + configName +  ".csv",true);

            //double u [] = t.generateBoWForImage(pathTest + "20170602_134259.png", "/Users/bdludwig/log_1.csv");
            //double v [] = t.generateBoWForImage(pathTest + "20170602_134148.png", "/Users/bdludwig/log_2.csv");

            //MP
            double uvComparisonResult = Math.round(myBgfwrdgen.compare(u,v)*1000000)/1000000.00;
            double wvComparisonResult = Math.round(myBgfwrdgen.compare(w,v)*1000000)/1000000.00;
            System.out.println("Comparison between u and v: " + uvComparisonResult);
            System.out.println("Comparison between w and v: " + wvComparisonResult);
            //System.out.println(t.compare(u, v));

            //MP
            PrintStream writeToResultFile = new PrintStream(new FileOutputStream(resultFile,true));
            writeToResultFile.print(PATH_TRAINING);
            writeToResultFile.print(";" + PATH_TEST);
            writeToResultFile.print(";" + cacheSize);
            writeToResultFile.print(";" + numWords);
            writeToResultFile.print(";" + useCanopies);
            writeToResultFile.print(";" + maxCandidates);
            writeToResultFile.print(";" + periodicPruning);
            writeToResultFile.print(";" + minDensity);
            writeToResultFile.print(";" + t2);
            writeToResultFile.print(";" + t1);
            writeToResultFile.print(";" + maxNumIterations);
            writeToResultFile.print(";" + fasterToString);
            writeToResultFile.print(";" + numSlots);
            writeToResultFile.print(";" + trainingExamplesProTuerschild);
            writeToResultFile.print(";" + (tuerschilder.length-wenigerTuerschilderDamitHeapNichtUeberlaeuft));
            writeToResultFile.print(";" + uvComparisonResult);
            writeToResultFile.print(";" + wvComparisonResult);
            writeToResultFile.print("\n");
            writeToResultFile.close();


            //MP
            System.out.println("Die Verarbeitung mit dem corpus-Format = " +
                    corpusFormat + ", dem query-Format = " + queryFormat + ", " +
                    "cacheSize = " + myBgfwrdgen.getCacheSize() + ", numWords = " +
                    myBgfwrdgen.getNumWords() + " und" + " Corpus-Groesse = " +
                    (tuerschilder.length-wenigerTuerschilderDamitHeapNichtUeberlaeuft) + " hat " +
                    "gedauert: " + (System.currentTimeMillis()-startTime)/60000 + " min.");
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
