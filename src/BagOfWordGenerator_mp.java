



import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;

import weka.clusterers.SimpleKMeans;
import weka.core.*;
import boofcv.alg.filter.binary.GThresholdImageOps;
import boofcv.alg.misc.ImageMiscOps;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;

public class BagOfWordGenerator_mp {
	//MP
	int counterMP = 0;

	private int numWords = 16; // Anzahl verschiedener Cluster, d.h. bag of words
	private int cacheSize = 16; // Anzahl Kacheln in X- und Y-Richtung des Bilds
	private Instances data; // Datensatz, der generiert werden soll
	private Instances codebook;
	private SimpleKMeans clusterer;	// Weka-Instanz des verwendeten Cluster-Algorithmus
	private String canopyOption = "";
	private int periodicPruning;
	private int maxCandidates;
	private int minDensity;
	private double t1,t2;
	private int maxNumIterations;
	private String fasterDistanceCalculations = "";
	private int numSlots;
	private String outputDebugInfo = "";
	
	public BagOfWordGenerator_mp(){
		this(16,16,
				false, 100,
				10000, 2, -1.0, -1.25,500,
				false,1,false);
	}
	public BagOfWordGenerator_mp(int cacheSize,int numWords){
		this(cacheSize,numWords,
				false, 100,
				10000, 2, -1.0, -1.25,500,
				false,1,false);
	}

	public BagOfWordGenerator_mp(int cacheSize,int numWords,
								 boolean useCanopiesForFasterClustering, int maxCandidates,
								 int periodicPruning, int minDensity, double t2, double t1,
								 int maxNumIterations, boolean faster,int numSlots,
								 boolean iWantDebugInfo) {
		this.numWords = numWords;
		this.cacheSize = cacheSize;
		if (useCanopiesForFasterClustering)
			canopyOption = " -C ";
		this.maxCandidates = maxCandidates;
		this.periodicPruning = periodicPruning;
		this.minDensity = minDensity;
		this.t2 = t2;
		this.t1 = t1;
		this.maxNumIterations = maxNumIterations;
		if (faster)
			fasterDistanceCalculations = " -fast";
		this.numSlots = numSlots;
		if (iWantDebugInfo)
			outputDebugInfo = " -output-debug-info";



		// konstruiere die Datenstruktur für WEKA
		// Vektor für die Werte eines Bilds.

		//MP:Ersatz für FastVector, vgl. https://stackoverflow.com/questions/26878103/the-type-fastvectore-is-deprecated
		ArrayList<Attribute> atts = new ArrayList<Attribute>();
		//for (int att = 0; att < numAtts; att++) {
		//	atts.add(new Attribute("Attribute" + att, att));
		//}

        //FastVector atts = new FastVector(cacheSize * cacheSize);
        int i = 0;
        
    	for (int y1 = 0; y1 < cacheSize; y1++) {
			for (int x1 = 0; x1 < cacheSize; x1++) {
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
		//System.out.println("Entered method 'greyValueVectorForCache' ..." + counterMP++);

		int i = 0;
		double [] vals = new double[cacheSize * cacheSize];
		double norm;
		int dim = 0;
		Instance inst;
		
		// Konstruiere pro Kachel einen normalisierten Vektor aus den
		// Graustufen aller Pixel im Cluster
		
		norm = 0; // Variable für Betrag des Vektors
		dim = 0; // Zähler durch alle Pixel der Kachel
		
		for (int y1 = 0; y1 < cacheSize; y1++) {
			for (int x1 = 0; x1 < cacheSize; x1++) {
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
		inst = new SparseInstance(cacheSize * cacheSize);

		//MP
        //System.out.println("The instance inst: " + inst);

		// Normalisiere den Vektor auf Betrag 1
		
		for (int k = 0; k < dim - 1; k++) {
			inst.setValue(data.attribute(k), vals[k] / norm);
		}
		
		inst.setValue(data.attribute(dim-1), vals[dim-1] / norm);
		
		return inst;
	}
	
	public int [] generateBoWForImage(String fileName, String logfile,boolean logHistograms) throws Exception {

		//MP
		System.out.println("Entered method 'generateBoWForImage' with file: " + fileName);
		counterMP = 0;

		BufferedImage image = UtilImageIO.loadImage(fileName);

		//"Converts a buffered image into an image of the specified type"
		//GrayF32: "Image with a pixel type of 32-bit float"
		GrayF32 color = ConvertBufferedImage.convertFromSingle(image, null, GrayF32.class);

		// konvertiere Foto in Graustufen nach der Sauvola-Methodik
		
		GrayU8 binary = new GrayU8(color.getWidth(), color.getHeight());
		GThresholdImageOps.localSauvola(color, binary, 8, 0.05f, true);
		
		/*
		GrayU8 toProcess = new GrayU8(binary.getWidth() + 1 + cacheSize/2,
				                      binary.getHeight() + cacheSize/2);
		*/

		//MP-Alternative zu oben (vgl. analog addTrainingExamples):
		int korrekturWidth = 0;
		int korrekturHeight = 0;
		if (binary.getWidth()% cacheSize != 0)
			korrekturWidth = cacheSize - binary.getWidth()% cacheSize;
		if (binary.getHeight()% cacheSize != 0)
			korrekturHeight = cacheSize - binary.getHeight()% cacheSize;
		GrayU8 toProcess = new GrayU8(binary.getWidth() + korrekturWidth,
				binary.getHeight() + korrekturHeight);

		ImageMiscOps.copy(0, 0, 0, 0,
				binary.getWidth(), binary.getHeight(),
				//ConvertBufferedImage.convertFromSingle(image, null, GrayU8.class),
				binary,
				toProcess);

		// Initalisiere Histogramm der Cluster

		int [] histogram = new int [numWords];
		for (int i = 0; i < numWords; i++) histogram[i] = 0;
		
		// konstruiere die einzelnen Kacheln
		
		Instance inst;

		// Schleife über das Bild, aber nicht pixelweise, sondern kachelweise
		
		for (int y = 0; y < toProcess.getHeight(); y += cacheSize) {
			for (int x = 0; x < toProcess.getWidth(); x += cacheSize) {
				// berechne normalisierten Vektor für aktuelle Kachel
				inst = greyValueVectorForCache(toProcess, x, y);
				
				// Bestimme den besten Cluster für das vorliegende Datenbeispiel
				int res = clusterer.clusterInstance(inst);
				
				// Erhöhe den Zähler für diesen Cluster in diesem Bild um 1
				
				histogram[res] ++;
			}
		}

		if (logHistograms){
			// Schreibe Log-Daten für dieses Bild

			PrintStream out = new PrintStream(new File(logfile));
			out.print(histogram[0]);
			for (int i = 1; i < numWords; i++) out.print(";" + histogram[i]);
			out.print("\n");
			out.close();
		}


		
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

		//MP mit Ludwig geeinigt, dass das weg kann.
		//image = VisualizeBinaryData.renderBinary(binary, false, null);

		/*MP: bei width ohne 1+C funktioniert's, bei height aber so und so nicht.
		GrayU8 toProcess = new GrayU8(binary.getWidth() +1+ cacheSize/2,
				                      binary.getHeight() + cacheSize/2);
	  	*/
		//MP-Alternative zu oben:
		int korrekturWidth = 0;
		int korrekturHeight = 0;
		if (binary.getWidth()% cacheSize != 0)
			korrekturWidth = cacheSize - binary.getWidth()% cacheSize;
		if (binary.getHeight()% cacheSize != 0)
			korrekturHeight = cacheSize - binary.getHeight()% cacheSize;
		GrayU8 toProcess = new GrayU8(binary.getWidth() + korrekturWidth,
				binary.getHeight() + korrekturHeight);


		//"Copies a rectangular region from one image into another."
		ImageMiscOps.copy(0, 0, 0, 0,
				binary.getWidth(), binary.getHeight(),
				//ConvertBufferedImage.convertFromSingle(image, null, GrayU8.class),
				binary,
				toProcess);
		
		Instance inst;
		
		for (int y = 0; y < toProcess.getHeight(); y += cacheSize) {
			for (int x = 0; x < toProcess.getWidth(); x += cacheSize) {
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
			// mit diesen Einstellungen konstruiert der Algorithmus numWords Cluster
			clusterer.setOptions(weka.core.Utils.splitOptions("-init 0 " + canopyOption +
					" -max-candidates " + maxCandidates +
					" -periodic-pruning " + periodicPruning +
					" -min-density " + minDensity +
					" -t2 " + t2 +
					" -t1 " + t1 +
					" -N " + numWords +
					" -A \"weka.core.EuclideanDistance -R first-last\" " +
					"-I " + maxNumIterations + fasterDistanceCalculations +
					" -num-slots " + numSlots +
					" -S 10" + outputDebugInfo));
			clusterer.buildClusterer(data);

			codebook = clusterer.getClusterCentroids();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
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

	//MP
	public int getCacheSize(){
		return cacheSize;
	}
	public int getNumWords(){
		return numWords;
	}
	public Instances getData(){return data;}
	

}