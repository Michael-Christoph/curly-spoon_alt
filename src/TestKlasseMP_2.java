import java.io.File;

/**
 * Created by Michael on 31.08.2017.
 */
public class TestKlasseMP_2 {
    public static void main(String[] args){
        long startTime = System.currentTimeMillis();
        BagOfWordGenerator_mp myBgfwrdgen = new BagOfWordGenerator_mp();
        String pathTraining = "corpus_jpg10vh/";
        String[] tuerschilder = new String[]{"pt_3-0-13","pt_3-0-26","pt_3-0-56","pt_3-0-57"};
        String pathTest = "query_jpg10vh/";

        try {
            //MP
            System.out.println(new File(".").getCanonicalPath());

            //MP
            String corpusFormat = "jpg";
            if (pathTest.indexOf('j') == -1)
                corpusFormat = "png";

            //MP-Versuch
            for (int sch = 0; sch<tuerschilder.length;sch++){
                for (int i=1; i<=18;i++){
                    myBgfwrdgen.addTrainingExample(pathTraining + tuerschilder[sch] + "/pic_" + Integer.toString(i) + "." + corpusFormat);
                }
            }



            //t.addTrainingExample(pathTraining + "PT_3.0.61/20170602_134243.png");

            myBgfwrdgen.cluster();

            //MP
            String queryFormat = "jpg";
            if (pathTest.indexOf('j') == -1)
                queryFormat = "png";

            double u[] = myBgfwrdgen.generateBoWForImage(pathTest + "pt_3-0-13/pic_19." + queryFormat,
                    "log_1.csv");
            double v[] = myBgfwrdgen.generateBoWForImage(pathTest + "pt_3-0-13/pic_20." + queryFormat,
                    "log_2.csv");
            double w[] = myBgfwrdgen.generateBoWForImage(pathTest + "pt_3-0-26/pic_19." + queryFormat,
                    "log_3.csv");

            //double u [] = t.generateBoWForImage(pathTest + "20170602_134259.png", "/Users/bdludwig/log_1.csv");
            //double v [] = t.generateBoWForImage(pathTest + "20170602_134148.png", "/Users/bdludwig/log_2.csv");

            //MP
            System.out.println("Comparison between u and v: " + myBgfwrdgen.compare(u, v));
            System.out.println("Comparison between w and v: " + myBgfwrdgen.compare(w,v));
            //System.out.println(t.compare(u, v));

            //MP
            System.out.println("Die Verarbeitung mit dem corpus-Format = " +
                    corpusFormat + ", dem query-Format = " + queryFormat + ", " +
                    "cacheSize = " + myBgfwrdgen.getCacheSize() + ", numWords = " + myBgfwrdgen.getNumWords() + " und" +
                    " Corpus-Groesse = " + tuerschilder.length + " hat " +
                    "gedauert: " + (System.currentTimeMillis()-startTime)/60000 + " min.");
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
