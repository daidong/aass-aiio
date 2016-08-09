package edu.ttu.aass.aiio;

import edu.ttu.aass.aiio.dataspace.YahooHDFSIterator;
import edu.ttu.aass.aiio.dataspace.YahooVectorIterator;
import org.nd4j.linalg.dataset.DataSet;

import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * Created by daidong on 8/3/16.
 */
public class AIIOMain {
	public static void main(String[] args) throws IOException {
		/*
		YahooHDFSIterator iter = new YahooHDFSIterator(args[0]);
		while (iter.hasNext()){
			String it = iter.nextSentence();
			System.out.println("Sentence: " + it);
		}
		*/

		YahooVectorIterator iter2 = new YahooVectorIterator(args[0], args[1], 32, 1000, 100);
		while (iter2.hasNext()){
			DataSet ds = iter2.next();
			System.out.println(ds.toString());
		}
	}
}
