/**
 * Created by Michael on 02.09.2017.
 */
public class BagOfWordsGenerator_trainedAndClustered {
    private BagOfWordsGenerator_mod bgfwrdgen;
    private String pathTraining;
    private String pathTest;
    private String[] schilder;
    private int trainingExamples;
    private String trialName;

    public BagOfWordsGenerator_trainedAndClustered(BagOfWordsGenerator_mod bgfwrdgen,
                                                   String pathTraining,
                                                   String pathTest,
                                                   String[] schilder,
                                                   int trainingExamples,
                                                   String trialName) throws Exception{
        this.bgfwrdgen = bgfwrdgen;
        this.pathTraining = pathTraining;
        this.pathTest = pathTest;
        this.schilder = schilder;
        this.trainingExamples = trainingExamples;
        this.trialName = trialName;

        trainAndCluster();
    }
    private void trainAndCluster() throws Exception{
        String corpusFormat = ".jpg";
        if (pathTest.indexOf('j') == -1)
            corpusFormat = ".png";
        for (int sch = 0; sch<schilder.length;sch++){
            for (int i=1; i<=trainingExamples;i++){
                bgfwrdgen.addTrainingExample(pathTraining + schilder[sch] + "/pic_" + Integer.toString(i) + corpusFormat);
            }
        }
        bgfwrdgen.cluster();
    }
    public int[] generateBoWForImage(String fileName) throws Exception{
        return bgfwrdgen.generateBoWForImage(fileName,"hierNichtRelevant",false);
    }
    public double compare(int [] v1, int [] v2){
        return bgfwrdgen.compare(v1,v2);
    }
    public int getTrainingExamples(){return trainingExamples;}
    public String[] getSchilder(){return schilder;}
    public String getPathTest(){return pathTest;}
    public String getPathTraining(){return pathTraining;}
}
