import java.io.FileOutputStream;
import java.io.PrintStream;

/**
 * Created by Michael on 02.09.2017.
 */
public class Tuerschilderkennung_eval {
    private static final String TRIAL_NAME = "eval_gesamt_xxx";
    private static final String PATH_TRAINING = "corpus_jpg15vh+autokorr+s10/";
    private static final String PATH_TEST = "query_jpg15vh+autokorr+s10/";
    private static final String[] SCHILDER = {"pt_3-0-13","pt_3-0-26",
            "pt_3-0-56","pt_3-0-57","pt_3-0-67","pt_3-0-68","pt_3-0-84a","pt_3-0-84b",
            "pt_3-0-84c","pt_3-0-84d"};
    private static final int NUM_TRAINING_EXAMPLES = 18;
    private static final String[] QUERY_PICS = {"/pic_19","/pic_20"};
    private static final int INIT_CACHE_SIZE = 4;
    private static final int MAX_CACHE_SIZE = 64;
    private static final int INIT_NUM_WORDS = 4;
    private static final int MAX_NUM_WORDS = 512;
    public static void main(String[] args) throws Exception{

        for (int cacheSize = INIT_CACHE_SIZE; cacheSize<= MAX_CACHE_SIZE; cacheSize *= 2){
            for (int numWords = INIT_NUM_WORDS; numWords<= MAX_NUM_WORDS; numWords *= 2){
                System.out.println("Entered new loop: cacheSize = " + cacheSize + ", numWords = " + numWords + ".");
                double fehlerRate = fehlerrate(cacheSize,numWords);

                System.out.println("Fehlerrate: " + fehlerRate);

                PrintStream writeToResultFile = new PrintStream(new FileOutputStream(TRIAL_NAME + ".csv",true));
                writeToResultFile.print(PATH_TRAINING);
                writeToResultFile.print(";" + PATH_TEST);
                writeToResultFile.print(";" + cacheSize);
                writeToResultFile.print(";" + numWords);
                writeToResultFile.print(";" + fehlerRate);
                writeToResultFile.print("\n");
                writeToResultFile.close();
            }
        }

    }
    private static double fehlerrate(int cacheSize, int numWords) throws Exception{
        BagOfWordGenerator_mp  bgfwrdgen = new BagOfWordGenerator_mp(cacheSize,
                numWords);
        BagOfWordGenerator_trainedAndClustered bfwrdgen_clustered =
                new BagOfWordGenerator_trainedAndClustered(bgfwrdgen,
                        PATH_TRAINING,PATH_TEST,SCHILDER,NUM_TRAINING_EXAMPLES,
                        TRIAL_NAME);
        KNNKlassifikator klassifikator = new KNNKlassifikator(bfwrdgen_clustered);
        int richtigeKlassifikationen = 0;
        int falscheKlassifikationen = 0;
        for (String schild: SCHILDER){
            for (String queryPic: QUERY_PICS){
                String classifiedAs = klassifikator.classify(schild+queryPic);
                if (classifiedAs.equals(schild))
                    richtigeKlassifikationen++;
                else
                    falscheKlassifikationen++;
            }
        }
        double fehlerRate = (double)falscheKlassifikationen/(double)(falscheKlassifikationen+richtigeKlassifikationen);
        return fehlerRate;
    }


}
