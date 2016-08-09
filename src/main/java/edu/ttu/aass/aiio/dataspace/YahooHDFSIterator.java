package edu.ttu.aass.aiio.dataspace;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;

/**
 * Created by daidong on 8/6/16.
 *
 * Generate iterator for Yahoo HDFS datasets.
 *
 * Current data format is:
 * #DATE \t hh:mm:ss,xxx \t INFO \t className \t ugi=MD5 \t ip=MD5 \t cmd=open \t src=xxx/xxx \t dst=xxx \t perm=null \n
 */
public class YahooHDFSIterator implements SentenceIterator, Iterable<String>{

	private BufferedReader reader;
	private InputStream backendStream;
	private long ts;
	private int words;

	public YahooHDFSIterator(String filePath) throws FileNotFoundException {
		this.backendStream = new FileInputStream(filePath);
		this.reader = new BufferedReader(new InputStreamReader(new BufferedInputStream(this.backendStream, 10 * 1024 * 1024)));
		this.ts = 0L;
		this.words = 0;
	}

	@Override
	public Iterator<String> iterator() {
		this.reset();
		Iterator<String> ret = new Iterator<String>() {
			@Override
			public boolean hasNext() {
				return YahooHDFSIterator.this.hasNext();
			}

			@Override
			public String next() {
				return YahooHDFSIterator.this.nextSentence();
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
				String fs[] = line.split("\t");
				String time = fs[0] + " " + fs[1];
				String fileId = fs[7];

				//2010-01-12	00:00:00,004
				String timePattern = "yyyy-MM-dd HH:mm:ss,SSS";
				SimpleDateFormat format = new SimpleDateFormat(timePattern);

				Date date = format.parse(time);
				long tmp = date.getTime();

				if (((tmp - ts) > 1000 && ts != 0) || words > 1000){
					this.ts = tmp;
					this.words = 0;
					return sb.toString();
				}

				sb.append(fileId + " ");
				this.words += 1;
				this.ts = tmp;
			}
			return null;
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ParseException e) {
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
			this.ts = 0L;
			this.words = 0;
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
