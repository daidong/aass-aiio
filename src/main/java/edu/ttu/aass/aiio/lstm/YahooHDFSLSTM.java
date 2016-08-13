package edu.ttu.aass.aiio.lstm;

import edu.ttu.aass.aiio.dataspace.YahooVectorIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.security.NoSuchAlgorithmException;
import java.util.Random;

/**
 * Created by daidong on 8/8/16.
 * We are using GravesLSTM to test this system
 */
public class YahooHDFSLSTM {

	public static YahooVectorIterator getYahooHDFSVectorizeIterator(
			String dataFile, String vectorizedFile, int miniBatchSize, int sequenceLength,
			int vecSize) throws IOException {
		File f1 = new File(dataFile);
		File f2 = new File(vectorizedFile);
		if (!f1.exists() || !f2.exists()){
			System.out.println("No data inputs or vectorlized data file");
			System.exit(0);
		}
		return new YahooVectorIterator(dataFile, vectorizedFile, miniBatchSize, sequenceLength, vecSize);
	}

	/**
	 * @param args
	 * @throws IOException
	 *
	 * args[0] = YahooHDFS Input Dataset
	 * args[1] = YahooHDFS Vector Representation (from Word2Vec)
	 * args[2] = YahooHDFS LSTM Network Serialization File
     */
	public static void main(String args[]) throws IOException, NoSuchAlgorithmException {

		DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

		String dataFile = args[0];
		String vectorizedFile = args[1];
		String lstmFile = args[2];

		int vecSize = 100; 							//Size of generated file vector. 100 by default
		/*
			Codes from example
		 */
		int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 32;						//Size of mini batch to use when  training
		int exampleLength = 1000;					//Length of each training example sequence to use. This could certainly be increased
		int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
		int numEpochs = 1;  						//Total number of training epochs

		MultiLayerNetwork net = null;

		if (!new File(lstmFile).exists()) {
			YahooVectorIterator iter = getYahooHDFSVectorizeIterator(dataFile,
					vectorizedFile, miniBatchSize, exampleLength, vecSize);

			int nOut = iter.totalOutcomes();

			/**
			 * Alex Black @AlexDBlack 23:17
			 right, probably have the wrong loss function then - mcxent assumes a one-hot vector for labels
			 try something like MSE

			 * Tunning options: http://deeplearning4j.org/lstm.html
			 * For LSTMs, use the softsign (not softmax) activation function over tanh (it’s faster and less prone to saturation (~0 gradients)).
			 * In general, stacking layers can help.
			 *
			 * MSE + softmax is not appropriate for either classification or regression
			 classification: MCXENT + softmax
			 regression: MSE + identity
			 also: no reason to mix softsign + tanh, use one or the other for both
			 */

			/*
			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
					.iterations(10)
					.learningRate(0.01)
					.rmsDecay(0.95)
					.seed(12345)
					.regularization(true)
					.l2(0.001)
					.weightInit(WeightInit.XAVIER)
					.updater(Updater.RMSPROP)
					.list()
					.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns())
							.nOut(lstmLayerSize)
							.activation("tanh").build())
					.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
							.activation("tanh")
							.build())
					.layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
							.activation("softsign")
							.nIn(lstmLayerSize)
							.nOut(nOut)
							.build())
					.backpropType(BackpropType.TruncatedBPTT)
					.tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
					.pretrain(false).backprop(true)
					.build();
			*/
			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
					.iterations(1)
					.learningRate(0.0018)
					.rmsDecay(0.95)
					.seed(12345)
					.regularization(true)
					.l2(1e-5)
					.weightInit(WeightInit.XAVIER)
					.updater(Updater.RMSPROP)
					.list()
					.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns())
							.nOut(lstmLayerSize)
							.activation("softsign").build())
					.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
							.activation("softsign")
							.build())
					.layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
							.activation("identity")
							.nIn(lstmLayerSize)
							.nOut(nOut)
							.build())
					.backpropType(BackpropType.TruncatedBPTT)
					.tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
					.pretrain(false).backprop(true)
					.build();

			net = new MultiLayerNetwork(conf);
			net.init();
			net.setListeners(new ScoreIterationListener(1));

			//Print the number of parameters in the network (and for each layer)
			Layer[] layers = net.getLayers();
			int totalNumParams = 0;
			for (int i = 0; i < layers.length; i++) {
				int nParams = layers[i].numParams();
				System.out.println("Number of parameters in layer " + i + ": " + nParams);
				totalNumParams += nParams;
			}
			System.out.println("Total number of network parameters: " + totalNumParams);


			//Do training, and then generate and print samples from network
			int miniBatchNumber = 0;
			for (int i = 0; i < numEpochs; i++) {
				while (iter.hasNext()) {
					DataSet ds = iter.next();
					if (ds == null)
						continue;
					net.fit(ds);
				}
				iter.reset();
			}

			System.out.println("\n\nTraining complete");

			ModelSerializer.writeModel(net, lstmFile, true);
		}

		net = ModelSerializer.restoreMultiLayerNetwork(lstmFile);

		/**
		 * Validate Prediction Accuracy
		 */
		YahooHDFSLSTMVerify verify = new YahooHDFSLSTMVerify(net, dataFile, vectorizedFile, vecSize);
		verify.runVerify();
	}
}