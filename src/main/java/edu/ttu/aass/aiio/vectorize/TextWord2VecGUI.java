package edu.ttu.aass.aiio.vectorize;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;
import java.nio.ByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;

/**
 * Created by daidong on 8/10/16.
 */
public class TextWord2VecGUI {

    private String vectorizedFile;
    private int vecSize;
    private HashMap<ByteBuffer, double[]> vecs;
    private WordVectors wordVectors;
    private BufferedReader vectorReader;

    public TextWord2VecGUI(String vecFile, int vecSize) throws IOException {
        this.vectorizedFile = vecFile;

        if (vecSize == 0) {
            this.wordVectors = WordVectorSerializer.loadGoogleModel(new File(this.vectorizedFile), true, false);
            this.vecSize = this.wordVectors.lookupTable().layerSize();
            this.vecs = null;

        } else {
            this.vecs = new HashMap<>();
            this.vecSize = vecSize;
            this.wordVectors = null;

            this.vectorReader = new BufferedReader(new FileReader(vectorizedFile));
            String line;
            while ((line = this.vectorReader.readLine()) != null) {
                String split[] = line.split(" ");
                double[] vec = new double[this.vecSize];
                for (int i = 0; i < this.vecSize; i++)
                    vec[i] = Double.parseDouble(split[1 + i]);
                vecs.put(ByteBuffer.wrap(split[0].getBytes()), vec);
            }
        }
    }

    private String ByteBuffer2String(ByteBuffer myByteBuffer){
        if (myByteBuffer.hasArray()) {
            return new String(myByteBuffer.array(),
                    myByteBuffer.arrayOffset() + myByteBuffer.position(),
                    myByteBuffer.remaining());
        } else {
            final byte[] b = new byte[myByteBuffer.remaining()];
            myByteBuffer.duplicate().get(b);
            return new String(b);
        }
    }

    public String[] predictNext(String inputs, int number) throws IOException, NoSuchAlgorithmException {
        inputs = inputs.toLowerCase();
        String[] words = inputs.split(" ");
        String[] predictedWords = new String[number];

        for (String line : words){
            String currFile = line;
            double[] crvector;

            if (this.vecs != null)
                crvector = this.vecs.get(ByteBuffer.wrap(currFile.getBytes()));
            else
                crvector = this.wordVectors.getWordVector(currFile);

            if (crvector == null)
                continue;

            if (this.vecs == null){
                int j = 0;
                for (String p : this.wordVectors.wordsNearest(currFile, number))
                    predictedWords[j++] = p;

            } else {

                double[] outputVector = crvector;

                double[] distance = new double[number];
                for (int j = 0; j < number; j++){
                    predictedWords[j] = "No Prediction";
                    distance[j] = Double.MAX_VALUE;
                }

                for (ByteBuffer word : this.vecs.keySet()) {
                    double[] v = this.vecs.get(word);
                    double dist = 0;
                    for (int j = 0; j < vecSize; j++)
                        dist += (Math.abs(v[j] - outputVector[j]) * Math.abs(v[j] - outputVector[j]));
                    dist = Math.sqrt(dist);

                    for (int j = 0; j < number; j++) {
                        if (dist < distance[j]) {
                            for (int k = number - 1; k > j; k--) {
                                distance[k] = distance[k - 1];
                                predictedWords[k] = predictedWords[k - 1];
                            }
                            distance[j] = dist;
                            predictedWords[j] = ByteBuffer2String(word);
                            break;
                        }
                    }
                }
            }
        }

        return predictedWords;
    }

    public void initGUI(){
        javax.swing.JFrame jf = new javax.swing.JFrame();
        jf.setTitle("Word2Vec English Word Test");
        jf.setSize(300,200);
        jf.setLocation(450,200);
        jf.setDefaultCloseOperation(3);
        jf.setResizable(true);

        java.awt.FlowLayout fl = new java.awt.FlowLayout();
        jf.setLayout(fl);

        javax.swing.JTextField jteName = new javax.swing.JTextField("", 20);
        jf.add(jteName);

        javax.swing.JLabel jlaName = new javax.swing.JLabel("Predicted Words...");
        jf.add(jlaName);

        javax.swing.JButton jbuName = new javax.swing.JButton("Predict");
        jf.add(jbuName);

        PredictActionListener pal = new PredictActionListener(jteName, jlaName);
        jbuName.addActionListener(pal);

        jf.setVisible(true);
    }

    public class PredictActionListener implements ActionListener{

        private javax.swing.JTextField jteName;
        private javax.swing.JLabel jlaName;

        public PredictActionListener(javax.swing.JTextField jteName, javax.swing.JLabel jlaName){
            this.jlaName = jlaName;
            this.jteName = jteName;
        }
        @Override
        public void actionPerformed(ActionEvent e) {
            String inputs = (String) jteName.getText();
            String[] predict = null;
            try {
                predict = predictNext(inputs, 10);
            } catch (IOException e1) {
                e1.printStackTrace();
            } catch (NoSuchAlgorithmException e1) {
                e1.printStackTrace();
            }
            String outputs = "<html>";
            for (String v : predict){
                outputs += (v + "<BR>");
            }
            outputs += "</html>";
            jlaName.setText(outputs);
        }
    }

    public static void main(String[] args) throws IOException {
        TextWord2VecGUI gui = new TextWord2VecGUI(args[0], 0);
        gui.initGUI();
    }
    
}
