import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by Michael on 16.08.2017.
 */
public class TestKlasseMP extends BagOfWordGenerator {

    public static void main (String[] args){

        Attribute attDropHeight = new Attribute("height");
        Attribute attQuality = new Attribute("quality");
        ArrayList<String> dropResults = new ArrayList<>();
        dropResults.add("broken");
        dropResults.add("notbroken");
        Attribute attDropResult = new Attribute("dropResult",dropResults);
        ArrayList<Attribute> atts = new ArrayList<>();
        atts.add(attDropHeight);
        atts.add(attQuality);
        atts.add(attDropResult);
        Instances breakingCups = new Instances("BreakingCups",atts,3);
        Instance cup1 = new DenseInstance(3);
        cup1.setValue(attDropHeight,2);
        cup1.setValue(attQuality,.8);
        cup1.setValue(attDropResult,"broken");
        Instance cup2 = new DenseInstance(3);
        cup2.setValue(attDropHeight,3);
        cup2.setValue(attQuality,0.9);
        cup2.setValue(attDropResult,"notbroken");

        breakingCups.add(cup1);
        breakingCups.add(cup2);


        System.out.println(breakingCups.toSummaryString());

    }
}
