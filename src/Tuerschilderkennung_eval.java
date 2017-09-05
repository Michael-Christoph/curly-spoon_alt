import java.io.FileOutputStream;
import java.io.PrintStream;

/**
 * Created by Michael on 02.09.2017.
 */
public class Tuerschilderkennung_eval {
    private static final String TRIAL_NAME = "eval_gesamt_nur-8bis16-256bis512-true";
    private static final String PATH_TRAINING = "corpus_jpg15vh+autokorr+s10/";
    private static final String PATH_TEST = "query_jpg15vh+autokorr+s10/";
    private static final String[] SCHILDER = {"pt_3-0-13","pt_3-0-26","pt_3-0-56","pt_3-0-57",
            "pt_3-0-67","pt_3-0-68","pt_3-0-84a","pt_3-0-84b","pt_3-0-84c","pt_3-0-84d"};
    private static final int NUM_TRAINING_EXAMPLES = 18;
    private static final String[] QUERY_PICS = {"/pic_19","/pic_20"};
    private static final int INIT_CACHE_SIZE = 8;
    private static final int MAX_CACHE_SIZE = 16;
    private static final int INIT_NUM_WORDS = 256;
    private static final int MAX_NUM_WORDS = 512;
    private static final boolean WEIGHTED_KNN = true;
    private static BagOfWordsGenerator_mod bgfwrdgen;
    public static void main(String[] args) throws Exception{

        for (int cacheSize = INIT_CACHE_SIZE; cacheSize<= MAX_CACHE_SIZE; cacheSize *= 2){
            for (int numWords = INIT_NUM_WORDS; numWords<= MAX_NUM_WORDS; numWords *= 2){

                System.out.println("Entered new loop: cacheSize = " + cacheSize + ", numWords = " + numWords + ".");

                double fehlerrate = ermittleFehlerrate(cacheSize,numWords);

                System.out.println("Fehlerrate: " + fehlerrate);

                //speichere die Konfiguration dieses Durchlaufs in einer Datei.
                PrintStream writeToResultFile = new PrintStream(new FileOutputStream(TRIAL_NAME + ".csv",
                        true));
                writeToResultFile.print(PATH_TRAINING);
                writeToResultFile.print(";" + PATH_TEST);
                writeToResultFile.print(";" + cacheSize);
                writeToResultFile.print(";" + numWords);
                writeToResultFile.print(";" + WEIGHTED_KNN);
                writeToResultFile.print(";" + fehlerrate);
                writeToResultFile.print("\n");
                writeToResultFile.close();
            }
        }
    }
    private static double ermittleFehlerrate(int cacheSize, int numWords) throws Exception{
        bgfwrdgen = new BagOfWordsGenerator_mod(cacheSize,
                numWords);
        fillAndClusterBagOfWordsGenerator();

        KNNKlassifikator klassifikator = new KNNKlassifikator(generateHistogramsForCorpus());
        int richtigeKlassifikationen = 0;
        int falscheKlassifikationen = 0;

        String queryFormat = setFormat(PATH_TEST);
        for (String schild: SCHILDER){
            for (String queryPic: QUERY_PICS){
                int[] queryHistogram = bgfwrdgen.generateBoWForImage(PATH_TEST + schild + queryPic + queryFormat,
                        "logHierNichtInteressant",false);
                String classifiedAs = klassifikator.classify(queryHistogram,SCHILDER,WEIGHTED_KNN);
                if (classifiedAs.equals(schild))
                    richtigeKlassifikationen++;
                else
                    falscheKlassifikationen++;
            }
        }
        double fehlerrate = (double)falscheKlassifikationen/(double)(falscheKlassifikationen+richtigeKlassifikationen);
        return fehlerrate;
    }
    private static void fillAndClusterBagOfWordsGenerator() throws Exception{
        String corpusFormat = setFormat(PATH_TRAINING);
        for (int sch = 0; sch<SCHILDER.length;sch++){
            for (int i=1; i<=NUM_TRAINING_EXAMPLES;i++){
                bgfwrdgen.addTrainingExample(PATH_TRAINING + SCHILDER[sch] +
                        "/pic_" + Integer.toString(i) + corpusFormat);
            }
        }
        bgfwrdgen.cluster();
    }
    private static int[][][] generateHistogramsForCorpus() throws Exception{
        int[][][] corpusHistograms;
        String trainingFormat = setFormat(PATH_TRAINING);
        corpusHistograms = new int[SCHILDER.length][][];
        for (int i=0;i<corpusHistograms.length;i++){
            corpusHistograms[i] = new int[NUM_TRAINING_EXAMPLES][];
            for (int j=0;j<corpusHistograms[i].length;j++){
                corpusHistograms[i][j] =
                        bgfwrdgen.generateBoWForImage(PATH_TRAINING +
                                SCHILDER[i] + "/pic_" + (j+1) + trainingFormat,
                                "logsHierNichtRelevant",false);
            }
        }
        return corpusHistograms;
    }
    private static String setFormat(String path){
        String dataFormat = ".jpg";
        if (path.indexOf('j') == -1)
            dataFormat = ".png";
        return dataFormat;
    }


}
