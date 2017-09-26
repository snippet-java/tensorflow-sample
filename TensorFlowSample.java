//jar@http://central.maven.org/maven2/org/tensorflow/tensorflow/1.3.0/tensorflow-1.3.0.jar;http://central.maven.org/maven2/org/tensorflow/libtensorflow/1.3.0/libtensorflow-1.3.0.jar;http://central.maven.org/maven2/org/tensorflow/libtensorflow_jni/1.3.0/libtensorflow_jni-1.3.0.jar
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.HttpClientBuilder;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/** Sample use of the TensorFlow Java API to label images using a pre-trained model. */
public class TensorFlowSample {

  public static void main(String[] args) {
  	TensorFlowSample test = new TensorFlowSample();
  	
    String modelDir = System.getProperty("user.dir");
    test.downloadJar("https://ibm.box.com/shared/static/tdadwaf74aoh5s4xcn2qm4ekt11ytlzb.txt", modelDir+File.separator+"imagenet_comp_graph_label_strings.txt");
    test.downloadJar("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/George_L._Burlingame_House%2C_1238_Harvard_St%2C_Houston_%28HDR%29.jpg/300px-George_L._Burlingame_House%2C_1238_Harvard_St%2C_Houston_%28HDR%29.jpg", modelDir+File.separator+"sample-image.jpg");
    test.downloadJar("https://ibm.box.com/shared/static/qzop56k0qf37mp5zq3q5wyr94ao87dct.pb", modelDir+File.separator+"tensorflow_inception_graph.pb");
    
    byte[] graphDef = readAllBytesOrExit(Paths.get(modelDir, "tensorflow_inception_graph.pb"));
    List<String> labels =
        readAllLinesOrExit(Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt"));
    byte[] imageBytes = readAllBytesOrExit(Paths.get(modelDir, "sample-image.jpg"));
    
    try (Tensor image = constructAndExecuteGraphToNormalizeImage(imageBytes)) {
      float[] labelProbabilities = executeInceptionGraph(graphDef, image);
      int bestLabelIdx = maxIndex(labelProbabilities);
      System.out.println(
          String.format(
              "BEST MATCH: %s (%.2f%% likely)",
              labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
    }
  }
  
  private void downloadJar(String url, String filename) {	
		System.out.println("Downloading file (" + filename + ")");
		
	    HttpClient httpClient = HttpClientBuilder.create().build();
		  HttpGet httpget = new HttpGet(url);
      HttpResponse response;
		try {
			response = httpClient.execute(httpget);
	        HttpEntity entity = response.getEntity();
	        
	        InputStream is = entity.getContent();
//	        ByteArrayOutputStream bos = new ByteArrayOutputStream();
	        FileOutputStream fos = new FileOutputStream(new File(filename));
	        int inByte;
	        while((inByte = is.read()) != -1)
//	        		bos.write(inByte);
	             fos.write(inByte);
	        is.close();
//	        byte[] output = bos.toByteArray();
//	        bos.close();
//	        return output;
	        fos.close();
	        
		} catch (ClientProtocolException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("File downloaded");
//		return null;
	}

  private static Tensor constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
    try (Graph g = new Graph()) {
      GraphBuilder b = new GraphBuilder(g);
      // Some constants specific to the pre-trained model at:
      // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
      //
      // - The model was trained with images scaled to 224x224 pixels.
      // - The colors, represented as R, G, B in 1-byte each were converted to
      //   float using (value - Mean)/Scale.
      final int H = 224;
      final int W = 224;
      final float mean = 117f;
      final float scale = 1f;

      // Since the graph is being constructed once per execution here, we can use a constant for the
      // input image. If the graph were to be re-used for multiple input images, a placeholder would
      // have been more appropriate.
      final Output input = b.constant("input", imageBytes);
      final Output output =
          b.div(
              b.sub(
                  b.resizeBilinear(
                      b.expandDims(
                          b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
                          b.constant("make_batch", 0)),
                      b.constant("size", new int[] {H, W})),
                  b.constant("mean", mean)),
              b.constant("scale", scale));
      try (Session s = new Session(g)) {
        return s.runner().fetch(output.op().name()).run().get(0);
      }
    }
  }

  private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
    try (Graph g = new Graph()) {
      g.importGraphDef(graphDef);
      try (Session s = new Session(g);
          Tensor result = s.runner().feed("input", image).fetch("output").run().get(0)) {
        final long[] rshape = result.shape();
        if (result.numDimensions() != 2 || rshape[0] != 1) {
          throw new RuntimeException(
              String.format(
                  "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                  Arrays.toString(rshape)));
        }
        int nlabels = (int) rshape[1];
        return result.copyTo(new float[1][nlabels])[0];
      }
    }
  }

  private static int maxIndex(float[] probabilities) {
    int best = 0;
    for (int i = 1; i < probabilities.length; ++i) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    return best;
  }

  private static byte[] readAllBytesOrExit(Path path) {
    try {
      return Files.readAllBytes(path);
    } catch (IOException e) {
      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
      System.exit(1);
    }
    return null;
  }

  private static List<String> readAllLinesOrExit(Path path) {
    try {
      return Files.readAllLines(path, Charset.forName("UTF-8"));
    } catch (IOException e) {
      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
      System.exit(0);
    }
    return null;
  }

  // In the fullness of time, equivalents of the methods of this class should be auto-generated from
  // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
  // like Python, C++ and Go.
  static class GraphBuilder {
    GraphBuilder(Graph g) {
      this.g = g;
    }

    Output div(Output x, Output y) {
      return binaryOp("Div", x, y);
    }

    Output sub(Output x, Output y) {
      return binaryOp("Sub", x, y);
    }

    Output resizeBilinear(Output images, Output size) {
      return binaryOp("ResizeBilinear", images, size);
    }

    Output expandDims(Output input, Output dim) {
      return binaryOp("ExpandDims", input, dim);
    }

    Output cast(Output value, DataType dtype) {
      return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
    }

    Output decodeJpeg(Output contents, long channels) {
      return g.opBuilder("DecodeJpeg", "DecodeJpeg")
          .addInput(contents)
          .setAttr("channels", channels)
          .build()
          .output(0);
    }

    Output constant(String name, Object value) {
      try (Tensor t = Tensor.create(value)) {
        return g.opBuilder("Const", name)
            .setAttr("dtype", t.dataType())
            .setAttr("value", t)
            .build()
            .output(0);
      }
    }

    private Output binaryOp(String type, Output in1, Output in2) {
      return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
    }

    private Graph g;
  }
}