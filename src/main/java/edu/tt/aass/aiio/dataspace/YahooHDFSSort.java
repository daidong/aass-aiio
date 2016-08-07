package edu.tt.aass.aiio.dataspace;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by daidong on 8/7/16.
 */
public class YahooHDFSSort {

	public static void main(String[] args) throws IOException {
		String inputFile = args[0];
		String outputFile = args[1];

		BufferedReader br = new BufferedReader(new InputStreamReader(new BufferedInputStream(new FileInputStream(inputFile), 10 * 1024 * 1024)));

		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new BufferedOutputStream(new FileOutputStream(outputFile), 10 * 1024 * 1024)));

		String line;
		while ((line = br.readLine()) != null){
			String fs[] = line.split("\t");
			String time = fs[0] + " " + fs[1];
			String fileId = fs[6];

			//2010-01-12	00:00:00,004
			String timePattern = "yyyy-MM-dd HH:mm:ss,SSS";
			SimpleDateFormat format = new SimpleDateFormat(timePattern);
			try {
				Date date = format.parse(time);
				long ts = date.getTime();

			} catch (ParseException e) {
				e.printStackTrace();
			}
		}

	}
}
