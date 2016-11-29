import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Prog3 {

    private static final int nData = 8125;
    private static final int trainSize = 6900;
    private static final int testSize = 1224;


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
        double[] trainY = new double[trainSize];
        double[][] trainX = new double[trainSize][23];

        double[] testY = new double[testSize];
        double[][] testX = new double[testSize][23];

        for (int i = 0; i < trainSize; i++) {
            trainY[i] = nOutput[i];

            for (int j = 0; j < 23; j++) {
                trainX[i][j] = (double) nInput[i][j] / 100;
            }
        }

        for (int i = 0; i < testSize; i++) {
            testY[i] = nOutput[i+trainSize];

            for (int j = 0; j < 23; j++) {
                testX[i][j] = (double) nInput[i+trainSize][j] / 100;
            }
        }


        // Init Weight
        double learnSpeed = 0.01;
        double[][] weight1 = new double[23][7];
        double[][] weight2 = new double[7][3];
        weightInit(weight1, 23, 7);
        weightInit(weight2, 7, 3);


        System.out.println("Hello World");

    }

    /**
     * Weight initialization between 0.1 - 0.9
     * @param weight weight to update with random number
     */
    private static void weightInit(double[][] weight, int iSize, int jSize) {
        Random random = new Random();
        for (int i = 0; i < iSize; i++) {
            for (int j = 0; j < jSize; j++) {
                weight[i][j] = 0.1 + (0.9-0.1) * random.nextDouble();
            }
        }
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

        try (BufferedReader br = new BufferedReader(new FileReader("mushroom.data"))) {
            String line = br.readLine();
            rawData[index] = line;
            index++;
            while (line != null) {
                line = br.readLine();
                rawData[index] = line;
                index++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return rawData;
    }

}
