import java.util.Arrays;

/**
 * Created by Michael on 02.09.2017.
 */
public class KNNKlassifikator {
    int[][][] corpusHistograms;

    public KNNKlassifikator(int[][][] corpusHistograms) throws Exception{
        this.corpusHistograms = corpusHistograms;
    }

    //queryPic wird demjenigen Türschild zugeordnet, dem die Mehrheit seiner k
    //nächsten Nachbar-pics zugehört.
    //k=numTrainingExamples, da ja zu hoffen ist, dass die k Trainings-Pics des richtigen
    //Türschilds besonders nah beim queryPic liegen.
    public String classify(int[] queryHistogram,String[] schilder,boolean weighted) throws Exception{

        //parallele String-double-Datenstruktur: Neighbor an Position x hat Comparison-Wert an Positon x.
        int k = corpusHistograms[0].length;
        String[] kNearestNeighbors = new String[k+1];//ein zusätzlicher Rang: nur für Verarbeitung (s. unten)
        double[] kNearestNumbers = new double[k+1];//ein zusätzlicher Rang: nur für Verarbeitung (s. unten)

        //durchlaufe die Schilder
        for (int i=0; i<corpusHistograms.length;i++){

            //durchlaufe die k pics eines Schildes
            for (int j=0; j<corpusHistograms[i].length;j++){
                double comparisonResult = compare(queryHistogram,corpusHistograms[i][j]);
                System.out.println("ComparisonResult: " + comparisonResult);

                //sortiere comparisonResult in die Rangliste ein.
                for (int rank=k;rank>0;rank--){
                    if (kNearestNumbers[rank-1]>=comparisonResult) {
                        squeezeInNumber(kNearestNumbers,comparisonResult,rank);
                        squeezeInNeighbor(kNearestNeighbors,schilder[i],rank);
                        break;
                    } else {
                        //springe einen Rang nach oben
                        if (rank-1 != 0) {
                            continue;
                        //weise Rang 1 zu und verlasse den Sortier-loop.
                        } else {
                            squeezeInNumber(kNearestNumbers,comparisonResult,rank-1);
                            squeezeInNeighbor(kNearestNeighbors,schilder[i],rank-1);
                            break;
                        }

                    }
                }
                System.out.println("kNearestNeighbors after comparison #" + (i*k + j + 1) + ": ");
                for (String neighbor: kNearestNeighbors)
                    System.out.println(neighbor);
                for (double num: kNearestNumbers)
                    System.out.println(num);
            }
        }
        String[] kNearestNeighborsWithoutBuffer = Arrays.copyOfRange(kNearestNeighbors,0,kNearestNeighbors.length-1);
        return majority(kNearestNeighborsWithoutBuffer,weighted);
    }
    private void squeezeInNeighbor(String[] currentKNearestNeighbors, String schild, int rank){
        String schildDasWeichenMuss = currentKNearestNeighbors[rank];
        currentKNearestNeighbors[rank] = schild;
        for (int i=rank+1;i<currentKNearestNeighbors.length;i++){
            String schildDasWeichenMuss2 = currentKNearestNeighbors[i];
            currentKNearestNeighbors[i] = schildDasWeichenMuss;
            schildDasWeichenMuss = schildDasWeichenMuss2;
        }
    }
    private void squeezeInNumber(double[] currentKNearestNumbers, double comparisonResult, int rank){
        double zahlDieWeichenMuss = currentKNearestNumbers[rank];
        currentKNearestNumbers[rank] = comparisonResult;
        for (int i=rank+1;i<currentKNearestNumbers.length;i++){
            double zahlDieWeichenMuss2 = currentKNearestNumbers[i];
            currentKNearestNumbers[i] = zahlDieWeichenMuss;
            zahlDieWeichenMuss = zahlDieWeichenMuss2;
        }
    }
    private String majority(String[] kNearestNeighbors,boolean weighted){
        String messageWeighted = "";
        if (weighted){
            int n = kNearestNeighbors.length;
            String[] kNearestNeighborsWeighted = new String[(n*(n+1))/2];
            int position = 0;
            for (int rank=0; rank<kNearestNeighbors.length;rank++){
                int numMultiplications = n-rank;
                for (int multiplier = 0; multiplier<numMultiplications;multiplier++){
                    kNearestNeighborsWeighted[position+multiplier] = kNearestNeighbors[rank];
                }
                position += numMultiplications;
            }
            kNearestNeighbors = Arrays.copyOfRange(kNearestNeighborsWeighted,0,kNearestNeighborsWeighted.length);
            messageWeighted = ", weighted";
        }
        System.out.println("Determined as kNearestNeighbors" + messageWeighted + ":");
        for (String neighbor: kNearestNeighbors)
            System.out.println(neighbor);

        String majorityName = "pt_0-0-0";
        int sizeMajority = 0;
        for (int i=0; i<kNearestNeighbors.length-1;i++){
            int size = 1;
            for (int m= i+1;m<kNearestNeighbors.length;m++){
                if (kNearestNeighbors[i].equals(kNearestNeighbors[m])){
                    size += 1;
                }
            }
            if (size>sizeMajority){
                majorityName = kNearestNeighbors[i];
                sizeMajority = size;
            }

        }
        System.out.println("Pic classified as " + majorityName + " with majority of " + sizeMajority +
            " out of " + (kNearestNeighbors.length-1) + " neighbors.");
        return majorityName;
    }
    //Vergleich der boW-Histogramme zweier pics
    public double compare(int [] v1, int [] v2) {
        double n1 = 0, n2 = 0, crossprod = 0;
        for (int i = 0; i < v1.length; i++) {
            crossprod += v1[i] * v2[i];
            n1 += v1[i]*v1[i];
            n2 += v2[i]*v2[i];
        }

        return crossprod/(Math.sqrt(n1)*Math.sqrt(n2));
    }
}
