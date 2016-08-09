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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
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

	public static void main(String args[]) throws IOException {

		String dataFile = args[0];
		String vectorizedFile = args[1];

		int vecSize = 100; 							//Size of generated file vector. 100 by default
		/*
			Codes from example
		 */
		int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 32;						//Size of mini batch to use when  training
		int exampleLength = 1000;					//Length of each training example sequence to use. This could certainly be increased
		int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
		int numEpochs = 1;							//Total number of training epochs
		int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch

		YahooVectorIterator iter = getYahooHDFSVectorizeIterator(dataFile,
				vectorizedFile, miniBatchSize, exampleLength, vecSize);

		int nOut = iter.totalOutcomes();

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(1)
				.learningRate(0.1)
				.rmsDecay(0.95)
				.seed(12345)
				.regularization(true)
				.l2(0.001)
				.list()
				.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns())
						.nOut(lstmLayerSize).updater(Updater.RMSPROP)
						.weightInit(WeightInit.DISTRIBUTION)
						.dist(new UniformDistribution(-0.08, 0.08))
						.activation("tanh").build())
				.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
						.updater(Updater.RMSPROP)
						.activation("tanh")
						.weightInit(WeightInit.DISTRIBUTION)
						.dist(new UniformDistribution(-0.08, 0.08))
						.build())
				.layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.activation("softmax")
						.nIn(lstmLayerSize)
						.nOut(nOut)
						.updater(Updater.RMSPROP)
						.weightInit(WeightInit.DISTRIBUTION)
						.dist(new UniformDistribution(-0.08, 0.08))
						.build())
				.backpropType(BackpropType.TruncatedBPTT)
				.tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
				.pretrain(false).backprop(true)
				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		//Print the number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);


		//Do training, and then generate and print samples from network
		int miniBatchNumber = 0;
		for( int i=0; i<numEpochs; i++ ){
			while(iter.hasNext()){
				DataSet ds = iter.next();
				if (ds == null)
					continue;

				net.fit(ds);
				if(++miniBatchNumber % generateSamplesEveryNMinibatches == 0){

					System.out.println("--------------------");
					System.out.println("Completed " + miniBatchNumber +
							" minibatches of size " + miniBatchSize +
							"x" + exampleLength + " characters" );

					/*
					String[] samples = sampleCharactersFromNetwork(
							generationInitialization,
							net,
							iter,
							rng,
							nCharactersToSample,
							nSamplesToGenerate);

					for( int j=0; j<samples.length; j++ ){
						System.out.println("----- Sample " + j + " -----");
						System.out.println(samples[j]);
						System.out.println();
					}
					*/
				}
			}

			iter.reset();
		}


		System.out.println("\n\nTraining complete");
	}
}