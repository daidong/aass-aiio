package edu.ttu.aass.aiio.lstm;

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
public class TextLSTMGUI {

    private BufferedReader reader;
    private MultiLayerNetwork model;
    private String wordsequence;
    private String vectorizedFile;
    private int vecSize;
    private HashMap<ByteBuffer, double[]> vecs;
    private BufferedReader vectorReader;

    public TextLSTMGUI(MultiLayerNetwork net, String file, String vecFile, int vecSize) throws IOException {
        this.model = net;
        this.wordsequence = file;
        this.vectorizedFile = vecFile;
        this.vecSize = vecSize;

        this.reader = new BufferedReader(new InputStreamReader(
                new BufferedInputStream(
                        new FileInputStream(this.wordsequence), 10 * 1024 * 1024)));

        this.vecs = new HashMap<>();
        this.vectorReader = new BufferedReader(new FileReader(vectorizedFile));
        String line;
        while ((line = this.vectorReader.readLine()) != null){
            String split[] = line.split(" ");
            double[] vec = new double[this.vecSize];
            for (int i = 0; i < this.vecSize; i++)
                vec[i] = Double.parseDouble(split[1 + i]);
            vecs.put(ByteBuffer.wrap(split[0].getBytes()), vec);
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
        this.model.rnnClearPreviousState();

        inputs = inputs.toLowerCase();
        String[] words = inputs.split(" ");
        INDArray output = null;

        for (String line : words){
            String currFile = line;
            ByteBuffer crbb = ByteBuffer.wrap(currFile.getBytes());
            double[] crvector = this.vecs.get(crbb);

            if (crvector == null)
                continue;

            INDArray input = Nd4j.create(crvector);
            output = model.rnnTimeStep(input);

        }

        double[] outputVector = output.data().asDouble();

        String[] predictedWords = new String[number];
        double[] distance = new double[number];
        for (int j = 0; j < number; j++){
            predictedWords[j] = "No Prediction";
            distance[j] = Double.MAX_VALUE;
        }

        for (ByteBuffer word : this.vecs.keySet()){
            double[] v = this.vecs.get(word);
            double dist = 0;
            for (int j = 0; j < vecSize; j++)
                dist += (Math.abs(v[j] - outputVector[j]) * Math.abs(v[j] - outputVector[j]));
            dist = Math.sqrt(dist);

            for (int j = 0; j < number; j++){
                if (dist < distance[j]){
                    for (int k = number - 1; k > j; k--) {
                        distance[k] = distance[k-1];
                        predictedWords[k] = predictedWords[k-1];
                    }
                    distance[j] = dist;
                    predictedWords[j] = ByteBuffer2String(word);
                    break;
                }
            }
        }

        return predictedWords;
    }

    public void initGUI(){
        javax.swing.JFrame jf = new javax.swing.JFrame();
        jf.setTitle("LSTM English Word Test");
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
    
}
