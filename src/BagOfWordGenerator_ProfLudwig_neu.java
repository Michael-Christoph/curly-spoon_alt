//package boofcv;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;

import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import boofcv.alg.filter.binary.GThresholdImageOps;
import boofcv.alg.misc.ImageMiscOps;
import boofcv.gui.binary.VisualizeBinaryData;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;

public class BagOfWordGenerator_ProfLudwig_neu {
	private final int NUM_WORDS = 512; // Anzahl verschiedener Cluster, d.h. bag of words
	private final int CACHE_SIZE = 16; // Anzahl Kacheln in X- und Y-Richtung des Bilds
	private Instances data; // Datensatz, der generiert werden soll
	private Instances codebook;
	private SimpleKMeans clusterer;	// Weka-Instanz des verwendeten Cluster-Algorithmus
	private double tfidf_matrix [][];
	
	public BagOfWordGenerator_ProfLudwig_neu() {
		// konstruiere die Datenstruktur für WEKA
		// Vektor für die Werte eines Bilds.
		
        FastVector atts = new FastVector(CACHE_SIZE * CACHE_SIZE);
        int i = 0;
        
    	for (int y1 = 0; y1 < CACHE_SIZE; y1++) {
			for (int x1 = 0; x1 < CACHE_SIZE; x1++) {
				// Konstruktion der Spalten im Datensatz
				atts.addElement(new Attribute("p_" + (i++)));
			}
		}
    	
    	data = new Instances("TestInstances", atts, 1000);
	}

	private Instance greyValueVectorForCache(GrayU8 toProcess, int x, int y) {
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
		
		inst = new SparseInstance(CACHE_SIZE * CACHE_SIZE);

		// Normalisiere den Vektor auf Betrag 1
		
		for (int k = 0; k < dim - 1; k++) {
			inst.setValue(data.attribute(k), vals[k] / norm);
		}
		
		inst.setValue(data.attribute(dim-1), vals[dim-1] / norm);
		
		return inst;
	}
	
	public double [] generateBoWForImage(String fileName, String logfile) throws Exception {
		BufferedImage image = UtilImageIO.loadImage(fileName);
		GrayF32 color = ConvertBufferedImage.convertFromSingle(image, null, GrayF32.class);
		int newX, newY;
		
		// konvertiere Foto in Graustufen nach der Sauvola-Methodik
		
		GrayU8 binary = new GrayU8(color.getWidth(), color.getHeight());
		GThresholdImageOps.localSauvola(color, binary, 8, 0.05f, true);
		
		if (binary.getWidth() % CACHE_SIZE == 0) newX = binary.getWidth();
		else newX = binary.getWidth() + (CACHE_SIZE - binary.getWidth() % CACHE_SIZE);
		
		if (binary.getHeight() % CACHE_SIZE == 0) newY = binary.getHeight();
		else newY = binary.getHeight() + (CACHE_SIZE - binary.getHeight() % CACHE_SIZE);
		
		GrayU8 toProcess = new GrayU8(newX, newY);
		ImageMiscOps.copy(0, 0, 0, 0, binary.getWidth(), binary.getHeight(), binary, toProcess);

		// Initalisiere Histogramm der Cluster

		double [] histogram = new double [NUM_WORDS];
		for (int i = 0; i < NUM_WORDS; i++) histogram[i] = 0;
		
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
		for (int i = 1; i < NUM_WORDS; i++) out.print("\t" + histogram[i]);
		out.print("\n");
		out.close();
		
		return histogram;
	}
	
	public void addTrainingExample(String fileName) throws FileNotFoundException {
		int newX, newY;
		
		// RGB-Bild in Graustufenbild konvertieren
		
		BufferedImage image = UtilImageIO.loadImage(fileName);
		GrayF32 color = ConvertBufferedImage.convertFromSingle(image, null, GrayF32.class);
		
		// konvertiere Foto in Graustufen nach der Sauvola-Methodik

		GrayU8 binary = new GrayU8(color.getWidth(), color.getHeight());
		GThresholdImageOps.localSauvola(color, binary, 8, 0.05f, true);
		
		if (binary.getWidth() % CACHE_SIZE == 0) newX = binary.getWidth();
		else newX = binary.getWidth() + (CACHE_SIZE - binary.getWidth() % CACHE_SIZE);
		
		if (binary.getHeight() % CACHE_SIZE == 0) newY = binary.getHeight();
		else newY = binary.getHeight() + (CACHE_SIZE - binary.getHeight() % CACHE_SIZE);
		
		GrayU8 toProcess = new GrayU8(newX, newY);
		ImageMiscOps.copy(0, 0, 0, 0, binary.getWidth(), binary.getHeight(), binary, toProcess);
		
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
			// mit diesen Einstellungen konstruiert der Algorithmus NUM_WORDS Cluster
			clusterer.setOptions(weka.core.Utils.splitOptions("-init 0 -max-candidates 100 -periodic-pruning 10000 -min-density 2.0 -t1 -1.25 -t2 -1.0 -N " + NUM_WORDS + " -A \"weka.core.EuclideanDistance -R first-last\" -I 500 -num-slots 1 -S 10"));
			clusterer.buildClusterer(data);

			codebook = clusterer.getClusterCentroids();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
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
		BagOfWordGenerator_ProfLudwig_neu t = new BagOfWordGenerator_ProfLudwig_neu();
		String pathTraining = "/Users/bdludwig/Software/Lire/liredemo/corpus_tuerschild/";
		String pathTest = "/Users/bdludwig/Software/Lire/liredemo/query_tuerschild/";
		
		try {
			t.addTrainingExample(pathTraining + "PT_3.0.61/20170602_134243.png");
			t.addTrainingExample(pathTraining + "PT_3.0.61/20170602_134244.png");
			t.addTrainingExample(pathTraining + "PT_3.0.61/20170602_134245.png");
			t.addTrainingExample(pathTraining + "PT_3.0.61/20170602_134245_1.png");
			t.addTrainingExample(pathTraining + "PT_3.0.61/20170602_134246.png");
			t.addTrainingExample(pathTraining + "PT_3.0.62/20170602_134307.png");
			t.addTrainingExample(pathTraining + "PT_3.0.62/20170602_134307_1.png");
			t.addTrainingExample(pathTraining + "PT_3.0.62/20170602_134309.png");
			t.addTrainingExample(pathTraining + "PT_3.0.62/20170602_134310.png");
			t.addTrainingExample(pathTraining + "kattenbeck/20170602_134205.png");
			t.addTrainingExample(pathTraining + "kattenbeck/20170602_134206.png");
			t.addTrainingExample(pathTraining + "kattenbeck/20170602_134207.png");
			t.addTrainingExample(pathTraining + "ludwig/20170602_134147.png");
			t.addTrainingExample(pathTraining + "ludwig/20170602_134148.png");
			t.addTrainingExample(pathTraining + "ludwig/20170602_134153.png");
			t.addTrainingExample(pathTraining + "ludwig/20170602_134154.png");
			t.addTrainingExample(pathTraining + "testothek/20170602_134258.png");
			t.addTrainingExample(pathTraining + "testothek/20170602_134259.png");
			t.addTrainingExample(pathTraining + "testothek/20170602_134259_1.png");
			t.addTrainingExample(pathTraining + "testothek/20170602_134300.png");
			
			t.cluster();
			
			double u [] = t.generateBoWForImage(pathTest + "20170602_134259.png", "/Users/bdludwig/log_1.csv");
			double v [] = t.generateBoWForImage(pathTest + "20170602_134148.png", "/Users/bdludwig/log_1.csv");
			
			System.out.println(t.compare(u, v));

			u = t.generateBoWForImage(pathTest + "20170602_134246.png", "/Users/bdludwig/log_1.csv");

			System.out.println(t.compare(u, v));

			u = t.generateBoWForImage(pathTest + "20170602_134209.png", "/Users/bdludwig/log_test.csv");

			System.out.println(t.compare(u, v));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}