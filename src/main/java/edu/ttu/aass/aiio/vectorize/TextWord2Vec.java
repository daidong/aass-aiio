package edu.ttu.aass.aiio.vectorize;

import edu.ttu.aass.aiio.dataspace.TextIterator;
import edu.ttu.aass.aiio.dataspace.YahooHDFSIterator;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.LowCasePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by daidong on 8/6/16.
 */
public class TextWord2Vec {

	private static Logger log = LoggerFactory.getLogger(TextWord2Vec.class);

	public static void main(String[] args) throws Exception {
		String file = args[0];
		String vec_outputs = args[1];

		log.info("Generating Text8 File Vecotr");

		SentenceIterator iter = new TextIterator(file);
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new LowCasePreProcessor());

		log.info("Traning model...");
		Word2Vec vec = new Word2Vec.Builder()
				.minWordFrequency(5)
				.iterations(10)
				.layerSize(100)
				.seed(42)
				.windowSize(15)
				.iterate(iter)
				.tokenizerFactory(t)
				.build();

		log.info("Fitting Word2Vec model...");
		vec.fit();

		// Write word vectors
		WordVectorSerializer.writeWordVectors(vec, vec_outputs);

		/*
		UiServer server = UiServer.getInstance();
		System.out.println("Started on port " + server.getPort());
		*/
	}
}
