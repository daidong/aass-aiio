package edu.ttu.aass.aiio.lstm;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.LowCasePreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.HashMap;

/**
 * Created by daidong on 8/10/16.
 */
public class YahooHDFSLSTMVerify {

    private BufferedReader reader;
    private MultiLayerNetwork model;
    private String hdfsLog;
    private String vectorizedFile;
    private int vecSize;
    private HashMap<ByteBuffer, double[]> vecs;
    private BufferedReader vectorReader;


    public YahooHDFSLSTMVerify(MultiLayerNetwork net, String file, String vecFile, int vecSize) throws IOException {
        this.model = net;
        this.hdfsLog = file;
        this.vectorizedFile = vecFile;
        this.vecSize = vecSize;

        this.reader = new BufferedReader(new InputStreamReader(
                new BufferedInputStream(
                        new FileInputStream(this.hdfsLog), 10 * 1024 * 1024)));

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

    public void runVerify() throws IOException {
        LowCasePreProcessor cp = new LowCasePreProcessor();
        String line;
        String predictedFile = "NONE";
        int correct = 0;
        int total = 0;

        this.model.rnnClearPreviousState();

        while ((line = reader.readLine()) != null){
            String fs[] = line.split("\t");
            String fileId = fs[7];
            String currFile = cp.preProcess(fileId);
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
            /*
            INDArray input = Nd4j.zeros(1, vecSize);
            for (int j = 0; j < vecSize; j++)
                input.putScalar(new int[]{0, j}, crvector[j]);
            */
            INDArray output = model.rnnTimeStep(input);

            double[] outputVector = output.data().asDouble();
            /*
            double[] outputVector = new double[vecSize];
            for (int j = 0; j < vecSize; j++)
                outputVector[j] = output.getDouble(0, j);
            */
            /*
                iterate all words to get the prediction
             */
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
