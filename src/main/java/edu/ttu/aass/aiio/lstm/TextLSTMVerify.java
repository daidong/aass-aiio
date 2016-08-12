package edu.ttu.aass.aiio.lstm;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.LowCasePreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;

/**
 * Created by daidong on 8/10/16.
 */
public class TextLSTMVerify {

    private BufferedReader reader;
    private MultiLayerNetwork model;
    private String wordsequence;
    private String vectorizedFile;
    private int vecSize;
    private HashMap<ByteBuffer, double[]> vecs;
    private BufferedReader vectorReader;


    public TextLSTMVerify(MultiLayerNetwork net, String file, String vecFile, int vecSize) throws IOException {
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

    public void runVerify() throws IOException, NoSuchAlgorithmException {
        String line;
        String predictedFile = "NONE";
        int correct = 0;
        int total = 0;

        this.model.rnnClearPreviousState();

        while ((line = reader.readLine()) != null){
            String currFile = line;
            ByteBuffer crbb = ByteBuffer.wrap(currFile.getBytes());
            double[] crvector = this.vecs.get(crbb);

            if (crvector == null)
                continue;

            total += 1;

            if (!predictedFile.equalsIgnoreCase("NONE")){
                if (currFile.equalsIgnoreCase(predictedFile)) {
                    System.out.println("Correctly predict current file access: " + currFile);
                    correct += 1;
                }
            }

            INDArray input = Nd4j.create(crvector);
            INDArray output = model.rnnTimeStep(input);

            double[] outputVector = output.data().asDouble();
            double MIN_DISTANCE = Double.MAX_VALUE;

            for (ByteBuffer file : this.vecs.keySet()){
                double[] v = this.vecs.get(file);
                double dist = 0;
                for (int j = 0; j < vecSize; j++)
                    dist += (Math.abs(v[j] - outputVector[j]) * Math.abs(v[j] - outputVector[j]));
                dist = Math.sqrt(dist);
                if (dist < MIN_DISTANCE){
                    MIN_DISTANCE = dist;
                    predictedFile = ByteBuffer2String(file);
                }
            }
        }
        System.out.println("Predict Ratio: " + ((double) correct / (double) total));
    }

}
