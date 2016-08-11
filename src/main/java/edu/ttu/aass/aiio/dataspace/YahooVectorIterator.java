package edu.ttu.aass.aiio.dataspace;

import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.LowCasePreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * Created by daidong on 8/8/16.
 */
public class YahooVectorIterator implements DataSetIterator {

	private BufferedReader reader;
	private InputStream backendStream;
	private long ts;

	private BufferedReader vectorReader;
	private int exampleLength;
	private int miniBatch;
	private Random random;
	private HashMap<ByteBuffer, double[]> vecs;
	private int vecSize;
	private int totalNumberFiles = 0;


	public YahooVectorIterator(String dataFile, String vectorizedFile,
							   int miniBatchSize, int sequenceLength,
							   int vecSize) throws IOException {
		this.exampleLength = sequenceLength;
		this.miniBatch = miniBatchSize;
		this.random = new Random(12345);

		this.backendStream = new FileInputStream(dataFile);
		this.reader = new BufferedReader(new InputStreamReader(new BufferedInputStream(this.backendStream, 10 * 1024 * 1024)));
		this.ts = 0L;
		this.vecSize = vecSize;

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
		this.totalNumberFiles = vecs.size();
	}

	@Override
	public boolean hasNext() {
		try {
			return reader.ready();
		} catch (Exception e) {
			return false;
		}
	}

	@Override
	public DataSet next() {
		return next(miniBatch);
	}

	@Override
	public DataSet next(int num) {
		INDArray input = Nd4j.create(new int[]{miniBatch,vecSize,exampleLength}, 'f');
		INDArray labels = Nd4j.create(new int[]{miniBatch,vecSize,exampleLength}, 'f');

		int currentBatchIdx = 0;
		int currentExampleIdx = 0;
		String previousFile = null;

		try {
			String line;
			LowCasePreProcessor cp = new LowCasePreProcessor();
			boolean skipWord = false;
			MessageDigest md = MessageDigest.getInstance("MD5");

			while ((line = reader.readLine()) != null){
				String fs[] = line.split("\t");
				String fileId = fs[7];
				byte[] md5File = md.digest(fileId.getBytes("UTF-8"));
				BigInteger bigInt = new BigInteger(1,md5File);
				String hashtext = bigInt.toString(16);
				while(hashtext.length() < 32 ){
					hashtext = "0"+hashtext;
				}
				String currFile = hashtext;

				if (currentBatchIdx == miniBatch){
					return new DataSet(input, labels);
				}

				if (previousFile != null){
					ByteBuffer pfbb = ByteBuffer.wrap(previousFile.getBytes());
					double[] pfvector = this.vecs.get(pfbb);

					ByteBuffer crbb = ByteBuffer.wrap(currFile.getBytes());
					double[] crvector = this.vecs.get(crbb);

					/**
					 * It is wired, even word2vec threshold is 0, there are still files that do not have vector
					 * representations. Need to check the reason.
					 * Now, we just skip those words.
					 */
					if (crvector != null) {
						for (int i = 0; i < vecSize; i++) {
							input.putScalar(new int[]{currentBatchIdx, i, currentExampleIdx - 1}, pfvector[i]);
							labels.putScalar(new int[]{currentBatchIdx, i, currentExampleIdx - 1}, crvector[i]);
						}
					} else {
						skipWord = true;
					}
				} else {
					ByteBuffer crbb = ByteBuffer.wrap(currFile.getBytes());
					double[] crvector = this.vecs.get(crbb);
					if (crvector == null)
						skipWord = true;
				}

				if (!skipWord) {
					previousFile = currFile;
					currentExampleIdx += 1;
				} else {
					skipWord = false;
					System.out.println("Skip Once");
				}

				if (currentExampleIdx == exampleLength){
					currentBatchIdx += 1;
					currentExampleIdx = 0;
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} catch (NoSuchAlgorithmException e) {
			e.printStackTrace();
		}

		return null;
	}

	@Override
	public int totalExamples() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public int inputColumns() {
		return this.vecSize;
	}

	@Override
	public int totalOutcomes() {
		return this.vecSize;
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public void reset() {
		try {
			if (backendStream instanceof FileInputStream) {
				((FileInputStream) backendStream).getChannel().position(0);
			} else backendStream.reset();
			reader = new BufferedReader(new InputStreamReader(new BufferedInputStream(backendStream, 10 * 1024 * 1024)));
			this.ts = 0L;
		} catch (Exception e){
			throw new RuntimeException(e);
		}
	}

	@Override
	public int batch() {
		return miniBatch;
	}

	@Override
	public int cursor() {
		return 0;
	}

	@Override
	public int numExamples() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException("Not implemented");
	}
}
