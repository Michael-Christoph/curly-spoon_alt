



import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;

import boofcv.gui.image.VisualizeImageData;
import boofcv.io.image.ConvertRaster;
import weka.clusterers.SimpleKMeans;
import weka.core.*;
import boofcv.alg.filter.binary.GThresholdImageOps;
import boofcv.alg.misc.ImageMiscOps;
import boofcv.gui.binary.VisualizeBinaryData;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;

public class BagOfWordGenerator {
	//MP
	int counterMP = 0;

	private final int NUM_WORDS = 64; // Anzahl verschiedener Cluster, d.h. bag of words
	private final int CACHE_SIZE = 16; // Anzahl Kacheln in X- und Y-Richtung des Bilds
	private Instances data; // Datensatz, der generiert werden soll
	private Instances codebook;
	private SimpleKMeans clusterer;	// Weka-Instanz des verwendeten Cluster-Algorithmus
	
	public BagOfWordGenerator() {
		// konstruiere die Datenstruktur für WEKA
		// Vektor für die Werte eines Bilds.

		//MP:Ersatz für FastVector, vgl. https://stackoverflow.com/questions/26878103/the-type-fastvectore-is-deprecated
		ArrayList<Attribute> atts = new ArrayList<Attribute>();
		//for (int att = 0; att < numAtts; att++) {
		//	atts.add(new Attribute("Attribute" + att, att));
		//}

        //FastVector atts = new FastVector(CACHE_SIZE * CACHE_SIZE);
        int i = 0;
        
    	for (int y1 = 0; y1 < CACHE_SIZE; y1++) {
			for (int x1 = 0; x1 < CACHE_SIZE; x1++) {
				// Konstruktion der Spalten im Datensatz
				//vgl. oben MP:Ersatz für FastVector
                atts.add(new Attribute("p_" + (i++)));

                //atts.addElement(new Attribute("p_" + (i++)));
			}
		}
    	
    	data = new Instances("TestInstances", atts, 1000);
	}

	private Instance greyValueVectorForCache(GrayU8 toProcess, int x, int y) {

		//MP
		System.out.println("Entered method 'greyValueVectorForCache' ..." + counterMP++);

		int i = 0;
		double [] vals = new double[CACHE_SIZE*CACHE_SIZE];
		double norm;
		int dim = 0;
		Instance inst;
		
		// Konstruiere pro Kachel einen normalisierten Vektor aus den
		// Graustufen aller Pixel im Cluster
		
		norm = 0; // Variable für Betrag des Vektors
		dim = 0; // Zähler durch alle Pixel der Kachel
		
		for (int y1 = 0; y1 < CACHE_SIZE; y1++) {
			for (int x1 = 0; x1 < CACHE_SIZE; x1++) {
				int p = toProcess.get(x + x1, y + y1);
				vals[dim++] = p; // speichere Graustufe
				norm += p*p; // Betrag zum Betrag
			}
		}
		
		// Betrag des Vektors
		
		norm = Math.sqrt(norm);
		if (norm == 0) norm = 1;

		// Erzeuge neues Datenbeispiel für WEKA
		
		//MP: Instance zu DenseInstance geaendert, wg. Ludwig:
		//"Wie ich gerade gesehen habe, ist Instance seit Weka 3.8
		// ein Interface. Wenn Sie in Zeile 76 DenseInstance oder
		// SparseInstance verwenden, sollte der Fehler verschwinden."
		inst = new SparseInstance(CACHE_SIZE*CACHE_SIZE);

		//MP
        //System.out.println("The instance inst: " + inst);

		// Normalisiere den Vektor auf Betrag 1
		
		for (int k = 0; k < dim - 1; k++) {
			inst.setValue(data.attribute(k), vals[k] / norm);
		}
		
		inst.setValue(data.attribute(dim-1), vals[dim-1] / norm);
		
		return inst;
	}
	
	public double [] generateBoWForImage(String fileName, String logfile) throws Exception {

		//MP
		System.out.println("Entered method 'generateBoWForImage' ...");
		counterMP = 0;

		BufferedImage image = UtilImageIO.loadImage(fileName);

		//"Converts a buffered image into an image of the specified type"
		//GrayF32: "Image with a pixel type of 32-bit float"
		GrayF32 color = ConvertBufferedImage.convertFromSingle(image, null, GrayF32.class);

		// konvertiere Foto in Graustufen nach der Sauvola-Methodik
		
		GrayU8 binary = new GrayU8(color.getWidth(), color.getHeight());
		GThresholdImageOps.localSauvola(color, binary, 8, 0.05f, true);
		
		/*
		GrayU8 toProcess = new GrayU8(binary.getWidth() + 1 + CACHE_SIZE/2,
				                      binary.getHeight() + CACHE_SIZE/2);
		*/

		//MP-Alternative zu oben (vgl. analog addTrainingExamples):
		int korrekturWidth = 0;
		int korrekturHeight = 0;
		if (binary.getWidth()%CACHE_SIZE != 0)
			korrekturWidth = CACHE_SIZE - binary.getWidth()%CACHE_SIZE;
		if (binary.getHeight()%CACHE_SIZE != 0)
			korrekturHeight = CACHE_SIZE - binary.getHeight()%CACHE_SIZE;
		GrayU8 toProcess = new GrayU8(binary.getWidth() + korrekturWidth,
				binary.getHeight() + korrekturHeight);

		ImageMiscOps.copy(0, 0, 0, 0,
				binary.getWidth(), binary.getHeight(),
				ConvertBufferedImage.convertFromSingle(image, null, GrayU8.class), toProcess);

		// Initalisiere Histogramm der Cluster

		double [] histogram = new double [NUM_WORDS];
		for (int i = 0; i < 64; i++) histogram[i] = 0;
		
		// konstruiere die einzelnen Kacheln
		
		Instance inst;

		// Schleife über das Bild, aber nicht pixelweise, sondern kachelweise
		
		for (int y = 0; y < toProcess.getHeight(); y += CACHE_SIZE) {
			for (int x = 0; x < toProcess.getWidth(); x += CACHE_SIZE) {		
				// berechne normalisierten Vektor für aktuelle Kachel
				inst = greyValueVectorForCache(toProcess, x, y);
				
				// Bestimme den besten Cluster für das vorliegende Datenbeispiel
				int res = clusterer.clusterInstance(inst);
				
				// Erhöhe den Zähler für diesen Cluster in diesem Bild um 1
				
				histogram[res] ++;
			}
		}

		// Schreibe Log-Daten für dieses Bild
		
		PrintStream out = new PrintStream(new File(logfile));
		out.print(histogram[0]);
		for (int i = 1; i < 64; i++) out.print("\t" + histogram[i]);
		out.print("\n");
		out.close();
		
		return histogram;
	}
	
	public void addTrainingExample(String fileName) throws FileNotFoundException {	
		// RGB-Bild in Graustufenbild konvertieren
		
		//"describes an Image with an accessible buffer of image data."
		BufferedImage image = UtilImageIO.loadImage(fileName);

		//"Converts a buffered image into an image of the specified type"
		//GrayF32: "Image with a pixel type of 32-bit float"
		GrayF32 color = ConvertBufferedImage.convertFromSingle(image, null, GrayF32.class);

		//GrayU8: "Creates a new gray scale (single band/color) image."
		GrayU8 binary = new GrayU8(color.getWidth(), color.getHeight());
		//"Applies Sauvola thresholding to the input image."
		GThresholdImageOps.localSauvola(color, binary, 8, 0.05f, true);

		//MP
		//BufferedImage imageBinary = VisualizeBinaryData.renderBinary(binary,false,null);

		//"Renders a binary image. 0 = black and 1 = white.
		//param invert: if true it will invert the image on output.
		image = VisualizeBinaryData.renderBinary(binary, false, null);

		/*MP: bei width ohne 1+C funktioniert's, bei height aber so und so nicht.
		GrayU8 toProcess = new GrayU8(binary.getWidth() +1+ CACHE_SIZE/2,
				                      binary.getHeight() + CACHE_SIZE/2);
	  	*/
		//MP-Alternative zu oben:
		int korrekturWidth = 0;
		int korrekturHeight = 0;
		if (binary.getWidth()%CACHE_SIZE != 0)
			korrekturWidth = CACHE_SIZE - binary.getWidth()%CACHE_SIZE;
		if (binary.getHeight()%CACHE_SIZE != 0)
			korrekturHeight = CACHE_SIZE - binary.getHeight()%CACHE_SIZE;
		GrayU8 toProcess = new GrayU8(binary.getWidth() + korrekturWidth,
				binary.getHeight() + korrekturHeight);


		//"Copies a rectangular region from one image into another."
		ImageMiscOps.copy(0, 0, 0, 0,
				binary.getWidth(), binary.getHeight(),
				ConvertBufferedImage.convertFromSingle(image, null, GrayU8.class),
				toProcess);
		
		Instance inst;
		
		for (int y = 0; y < toProcess.getHeight(); y += CACHE_SIZE) {
			for (int x = 0; x < toProcess.getWidth(); x += CACHE_SIZE) {		
				inst = greyValueVectorForCache(toProcess, x, y);
				data.add(inst);
			}
		}
	}
	
	// Diese Methode ruft den gewählten Cluster-Algorithmus von WEKA auf.
	
	public void cluster() {
		System.out.println("# training instances: " + data.numInstances());
		try {
			// wir benutzen hier einen einfachen KMeans-Algorithmus
			clusterer = new SimpleKMeans();
			// die genauen Einstellungen können von der WEKA-GUI übernommen werden
			// mit diesen Einstellungen konstruiert der Algorithmus 64 Cluster
			clusterer.setOptions(weka.core.Utils.splitOptions("-init 0 -max-candidates 100 -periodic-pruning 10000 -min-density 2.0 -t1 -1.25 -t2 -1.0 -N 64 -A \"weka.core.EuclideanDistance -R first-last\" -I 500 -num-slots 1 -S 10"));
			clusterer.buildClusterer(data);

			codebook = clusterer.getClusterCentroids();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	//Vergleich der boW-Histogramme zweier pics
	public double compare(double [] v1, double [] v2) {
		double n1 = 0, n2 = 0, crossprod = 0;
		for (int i = 0; i < v1.length; i++) {
			crossprod += v1[i] * v2[i];
			n1 += v1[i]*v1[i];
			n2 += v2[i]*v2[i];
		}
		
		return crossprod/(Math.sqrt(n1)*Math.sqrt(n2));
	}
	
	public static void main (String [] args) {
		BagOfWordGenerator t = new BagOfWordGenerator();
		String pathTraining = "corpus/";
		String pathTest = "query/";

		try {
			//MP
		    System.out.println(new File(".").getCanonicalPath());
			//MP-Versuch
			for (int i=1; i<=3;i++){
				t.addTrainingExample(pathTraining + "pt_3-0-13/pic_" + Integer.toString(i) + ".jpg");
			}
			//t.addTrainingExample("\\Users\\Michael\\iss-java-projekt\\src\\corpus\\Krypto-Buch\\pic_1.png");
			//t.addTrainingExample("corpus\\Krypto-Buch\\pic_1.png");
			//t.addTrainingExample(pathTraining + "Krypto-Buch/pic_" + Integer.toString(2)+ ".jpg");


			//t.addTrainingExample(pathTraining + "PT_3.0.61/20170602_134243.png");
			
			t.cluster();
			
			double u[] = t.generateBoWForImage(pathTest + "pt_3-0-13/pic_19.jpg",
					"log_1.csv");
			double v[] = t.generateBoWForImage(pathTest + "pt_3-0-13/pic_20.jpg",
					"log_2.csv");
			//double u [] = t.generateBoWForImage(pathTest + "Krypto-Buch/pic_19.jpg", "log_1.csv");
			//double v [] = t.generateBoWForImage(pathTest + "Krypto-Buch/pic_20.jpg", "log_2.csv");
			//double u [] = t.generateBoWForImage(pathTest + "20170602_134259.png", "/Users/bdludwig/log_1.csv");
			//double v [] = t.generateBoWForImage(pathTest + "20170602_134148.png", "/Users/bdludwig/log_2.csv");

			//MP
			System.out.println("Comparison between u and v:" + t.compare(u, v));
			//System.out.println(t.compare(u, v));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}