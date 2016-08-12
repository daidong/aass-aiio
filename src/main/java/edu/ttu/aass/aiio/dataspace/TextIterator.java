package edu.ttu.aass.aiio.dataspace;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import java.io.*;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;

/**
 * Created by daidong on 8/6/16.
 *
 * Generate iterator for text8 datasets. This is an example dataset from Google/Work2Vec source code.
 *
 */
public class TextIterator implements SentenceIterator, Iterable<String>{

	private BufferedReader reader;
	private InputStream backendStream;
	private int words;

	public TextIterator(String filePath) throws FileNotFoundException {
		this.backendStream = new FileInputStream(filePath);
		this.reader = new BufferedReader(new InputStreamReader(new BufferedInputStream(this.backendStream, 10 * 1024 * 1024)));
		this.words = 0;
	}

	@Override
	public Iterator<String> iterator() {
		this.reset();
		Iterator<String> ret = new Iterator<String>() {
			@Override
			public boolean hasNext() {
				return TextIterator.this.hasNext();
			}

			@Override
			public String next() {
				return TextIterator.this.nextSentence();
			}

			@Override
			public void remove() {
				throw new UnsupportedOperationException();
			}
		};
		return ret;
	}

	@Override
	public String nextSentence() {
		try {
			StringBuilder sb = new StringBuilder();
			String line;
			while ((line = reader.readLine()) != null){

				if (words > 10000){
					this.words = 0;
					return sb.toString();
				}

				sb.append(line + " ");
				this.words += 1;
			}
			return null;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
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
	public void reset() {
		try {
			if (backendStream instanceof FileInputStream) {
				((FileInputStream) backendStream).getChannel().position(0);
			} else backendStream.reset();
			reader = new BufferedReader(new InputStreamReader(new BufferedInputStream(backendStream, 10 * 1024 * 1024)));
		} catch (Exception e){
			throw new RuntimeException(e);
		}
	}

	@Override
	public void finish() {
		try {
			if (backendStream != null) backendStream.close();
			if (reader != null) reader.close();
		} catch (Exception e) {
			// do nothing here
		}
	}

	@Override
	public SentencePreProcessor getPreProcessor() {
		return null;
	}

	@Override
	public void setPreProcessor(SentencePreProcessor preProcessor) {
	}
}
