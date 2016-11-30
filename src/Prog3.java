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

//        String[] output = new String[nData-1];
        int[] nOutput = new int[nData-1];
        int[][] nInput = new int[nData-1][22];

        // Separate data to input and output, then normalized them to integer representation
        separate(rawData, nOutput, nInput);

        // Divide into train set (6900) and test set (1224)
        double[] trainY = new double[trainSize];
        double[][] trainX = new double[trainSize][22];

        double[] testY = new double[testSize];
        double[][] testX = new double[testSize][22];

        divideTrainingTest(trainY, trainX, testY, testX, nOutput, nInput);

        // Init Weight
        double learnRate = 0.005;
        double[][] weight1 = new double[22][7];
        double[][] weight2 = new double[7][3];
        double[][] weight3 = new double[3][1];
        weightInit(weight1, 22, 7);
        weightInit(weight2, 7, 3);
        weightInit(weight3, 3, 1);

        // Hidden Layer
        double[] h1 = new double[7];
        double[] h2 = new double[3];
        double output;
        double temp = 0.0;
        int count = 0;
        double accurate;
        double delta1;
        double[] delta2 = new double[3];
        double[] delta3 = new double[7];


        // Calculate Back Propagation
        for (int l = 0; l < 10000; l++) {
            for (int k = 0; k < trainSize; k++) {
                // h1
                for (int i = 0; i < 7; i++) {
                    for (int j = 0; j < 22; j++) {
                        temp += trainX[k][j] * weight1[j][i];
                    }
                    temp = sigmoid(temp);
                    h1[i] = temp;
                    temp = 0;
                }

                // h2
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 7; j++) {
                        temp += h1[j] * weight2[j][i];
                    }
                    temp = sigmoid(temp);
                    h2[i] = temp;
                    temp = 0;
                }

                // output
                for (int i = 0; i < 3; i++) {
                    temp += h2[i] * weight3[i][0];
                }
                temp = sigmoid(temp);
                output = temp;
                temp = 0;

                // Back Propagation
                double check;
                if (output < 0.5)
                    check = 0.0;
                else
                    check = 1.0;

                if (check == trainY[k]) {
                    count++;
                } else { // Perform Back Propagation
                    // delta1 and weight1
                    delta1 = output * (1 - output) * (trainY[k] - output);
                    for (int i = 0; i < 3; i++) {
                        weight3[i][0] = weight3[i][0] + (learnRate * delta1 * output);
                    }

                    // delta2 and weight2
                    for (int i = 0; i < 3; i++) {
                        delta2[i] = h2[i] * (1 - h2[i]) * (delta1 * weight3[i][0]);
                        for (int j = 0; j < 7; j++) {
                            weight2[j][i] = weight2[j][i] + (learnRate * delta2[i] * h2[i]);
                        }
                    }

                    // delta3 and weight3
                    double sum = 0;
                    for (int i = 0; i < 7; i++) {
                        for (int j = 0; j < 3; j++) {
                            sum += delta2[j] * weight2[i][j];
                        }
                        delta3[i] = h1[i] * (1 - h1[i]) * sum;
                        for (int j = 0; j < 22; j++) {
                            weight1[j][i] = weight1[j][i] + (learnRate * delta3[i] * h1[i]);
                        }
                    }
                }
            }
            accurate = (double) count / (double) trainSize;
            count = 0;
            if (l % 100 == 0)
                System.out.println("Epoch " + l + " = " +  accurate);
        }
        System.out.println("Hello World");

    }

    /**
     * Divide data into training data and testing data
     * @param trainY train output Y
     * @param trainX train input X
     * @param testY test output Y
     * @param testX test input X
     * @param nOutput original output Y
     * @param nInput original output X
     */
    private static void divideTrainingTest (double[] trainY, double[][] trainX, double[] testY, double[][] testX, int[] nOutput, int[][] nInput) {
        for (int i = 0; i < trainSize; i++) {
            trainY[i] = nOutput[i];
            for (int j = 0; j < 22; j++) {
                trainX[i][j] = (double) nInput[i][j] / 100;
            }
        }

        for (int i = 0; i < testSize; i++) {
            testY[i] = nOutput[i+trainSize];
            for (int j = 0; j < 22; j++) {
                testX[i][j] = (double) nInput[i+trainSize][j] / 100;
            }
        }

    }

    /**
     * Separate rawData into integer representation of features and output
     * @param rawData Original datas
     * @param nOutput Output Y
     * @param nInput Input features X
     */
    private static void separate(String[] rawData, int[] nOutput, int[][] nInput) {
        for (int i = 0; i < nData-1; i++) {
            String[] data = rawData[i].split(",");

            // 1 is eatable, 0 is poisonous
            if (Objects.equals(data[0], "e"))
                nOutput[i] = 1;
            else
                nOutput[i] = 0;
            //  output[i] = data[0];

            // Constructing input feature of 8124x23
            for (int j = 1; j < 23; j++) {
                char c = data[j].charAt(0);
                nInput[i][j-1] = c;
            }
        }
    }

    /**
     * Weight initialization between 0.1 - 0.9
     * @param weight weight to update with random number
     */
    private static void weightInit(double[][] weight, int iSize, int jSize) {
        Random random = new Random();
        for (int i = 0; i < iSize; i++) {
            for (int j = 0; j < jSize; j++) {
//                weight[i][j] = 0.1 + (0.9-0.1) * random.nextDouble();
                weight[i][j] = -0.9 + (0.9+0.9) * random.nextDouble();
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
