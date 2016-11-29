import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Objects;

public class Prog3 {

    private static final int nData = 8125;
    private static final int trainSize = 6000;
    private static final int testSize = 2124;

    public static void  main (String[] args) throws IOException{

        String[] rawData;
        rawData = readFile();

        String[] output = new String[nData-1];
        int[] nOutput = new int[nData-1];
        int[][] nInput = new int[nData-1][23];

        // Separate data to input and output, then normalized them to integer representation
        for (int i = 0; i < nData-1; i++) {
            String[] data = rawData[i].split(",");

            // 1 is eatable, 0 is poisonous
            if (Objects.equals(data[0], "e"))
                nOutput[i] = 1;
            else
                nOutput[i] = 0;
            output[i] = data[0];

            // Constructing input feature of 8124x23
            for (int j = 0; j < 23; j++) {
                char c = data[j].charAt(0);
                nInput[i][j] = c;
            }
        }


        // Divide into train set (6000) and test set (2124)
        int[] trainY = new int[trainSize];
        int[][] trainX = new int[trainSize][23];

        int[] testY = new int[testSize];
        int[][] testX = new int[testSize][23];

        for (int i = 0; i < trainSize; i++) {
            trainY[i] = nOutput[i];

            for (int j = 0; j < 23; j++) {
                trainX[i][j] = nInput[i][j];
            }
        }

        for (int i = 0; i < testSize; i++) {
            testY[i] = nOutput[i+trainSize];

            for (int j = 0; j < 23; j++) {
                testX[i][j] = nInput[i+trainSize][j];
            }
        }


        //
        double learnSpeed = 0.01;

        double number = 0.755;

        double result = sigmoid(number);

        System.out.println("Hello World");

    }


    /**
     * Sigmoid function
     * @param num a number to convert through sigmoid
     * @return a double number range between 0 to 1
     */
    private static double sigmoid(double num) {
        return 1 / (1 + Math.pow(Math.E, (-1 * num)));
    }

    /**
     * Read file "mushroom.data"
     * @return String representation of the file line by line
     * @throws IOException Input/Output, file not found exception
     */
    private static String[] readFile() throws IOException {
        String[] rawData = new String[nData];
        int index = 0;

        BufferedReader br = new BufferedReader(new FileReader("mushroom.data"));
        try {
            StringBuilder sb = new StringBuilder();
            String line = br.readLine();
            rawData[index] = line;
            index++;
            while (line != null) {
                sb.append(line);
                sb.append(System.lineSeparator());
                line = br.readLine();
                rawData[index] = line;
                index++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            br.close();
        }
        return rawData;
    }

}
