/**
 * Sereyvathanak Khorn
 * CSCI 402
 * Program 3
 *
 * ANN Back Propagation on mushroom data to determine if a mushroom is poisonous or eatable.
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class Prog3 {

    private static final int nData = 8125;              // Data size
    private static final int trainSize = 6900;          // train data size - 85%
    private static final int testSize = 1224;           // test data size - 15%
    private static double learnRate = 0.001;     // Learn rate
    private static final int stop = 5;                  // Count for training to stop early
    private static final int max = 1000000;               // MAX Epoch iteration


    /**
     * Program main
     * @param args arguments
     * @throws IOException Input/Output exception
     */
    public static void main (String[] args) throws IOException{
        String[] rawData;
        rawData = readFile();
        randomizedData(rawData);

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
        double[][] weight0 = new double[22][15];
        double[][] weight1 = new double[15][7];
        double[][] weight2 = new double[7][3];
        double[][] weight3 = new double[3][1];
        weightInit(weight0, 22, 15);
        weightInit(weight1, 15, 7);
        weightInit(weight2, 7, 3);
        weightInit(weight3, 3, 1);
        double accuracy;
        double accuracyTest;

        // Start time
        long startTime = System.currentTimeMillis();

        // Perform ANN Back Propagation
        accuracy = ANN(weight0, weight1, weight2, weight3, trainX, trainY);

        // Print the weights
        printWeight(weight0, weight1, weight2, weight3);

        // Training data
        System.out.println("\nTraining accuracy: " + accuracy);
        // Testing data
        accuracyTest = testAccurate(weight0, weight1, weight2, weight3, testX, testY);
        System.out.println("Testing accuracy: " + accuracyTest);

        // End time
        long endTime = System.currentTimeMillis();
        System.out.println("Total time: " + ((endTime - startTime) / 1000) + " seconds");

        fileWrite(weight0, "weight0.txt");
        fileWrite(weight1, "weight1.txt");
        fileWrite(weight2, "weight2.txt");
        fileWrite(weight3, "weight3.txt");
    }


    /**
     * Create a file to represent weight
     * @param weight weight
     */
    private static void fileWrite(double[][] weight, String filename) {
        try{
            PrintWriter writer = new PrintWriter(filename, "UTF-8");
            for (double[] aWeight1 : weight) {
                for (double anAWeight1 : aWeight1) {
                    writer.printf("%6.3f ", anAWeight1);
                }
                writer.println();
            }
            writer.close();
        } catch (IOException e) {
            System.out.println("Fail to create a new file!");
        }
    }

    private static void loadWeight() {

    }

    /**
     * Shuffle order of an array
     * @param data array of data to be shuffled
     */
    private static void randomizedData(String[] data) {
        Random random = new Random();
        int num = 2000;
        int index1;
        int index2;
        String temp;

        for (int i = 0; i < num; i++) {
            index1 = random.nextInt(data.length-1);
            index2 = random.nextInt(data.length-1);

            temp = data[index1];
            data[index1] = data[index2];
            data[index2] = temp;
        }
    }

    /**
     * Perform accurate test on the test set
     * @param weight1 w1
     * @param weight2 w2
     * @param weight3 w3
     * @param testX test features X
     * @param testY test output Y
     */
    private static double testAccurate(double[][] weight0, double[][] weight1, double[][] weight2, double[][] weight3, double[][] testX, double[] testY) {
        double[] h0 = new double[15];
        double[] h1 = new double[7];
        double[] h2 = new double[3];
        double output;
        double temp = 0.0;
        double accurate;
        int count = 0;

        for (int k = 0; k < testSize; k++) {
            // h0
            for (int i = 0; i < 15; i++) {
                for (int j = 0; j < 22; j++) {
                    temp += testX[k][j] * weight0[j][i];
                }
                h0[i] = sigmoid(temp);
                temp = 0;
            }

            // h1
            for (int i = 0; i < 7; i++) {
                for (int j = 0; j < 15; j++) {
                    temp += h0[j] * weight1[j][i];
                }
                h1[i] = sigmoid(temp);
                temp = 0;
            }

            // h2
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 7; j++) {
                    temp += h1[j] * weight2[j][i];
                }
                h2[i] = sigmoid(temp);
                temp = 0;
            }

            // output
            for (int i = 0; i < 3; i++) {
                temp += h2[i] * weight3[i][0];
            }
            output = sigmoid(temp);
            temp = 0;

            // Normalized output to either 1 or 0
            double check;
            if (output < 0.5)
                check = 0.0;
            else
                check = 1.0;

            // Check data
            if (check == testY[k]) {
                count++;
            }
        }
        accurate = (double) count / (double) testSize;
        return accurate;
    }


    /**
     * Perform ANN Back Propagation
     * @param weight1 w1
     * @param weight2 w2
     * @param weight3 w3
     * @param trainX training features X
     * @param trainY training output Y
     * @return The accuracy of the training set
     */
    private static double ANN(double[][] weight0, double[][] weight1, double[][] weight2, double[][] weight3, double[][] trainX, double[] trainY) {
        // Hidden Layer
        double[] h0 = new double[15];
        double[] h1 = new double[7];
        double[] h2 = new double[3];
        double output;
        double temp = 0.0;
        double prevAccurate = 0;
        double delta1;
        double[] delta2 = new double[3];
        double[] delta3 = new double[7];
        double[] delta4 = new double[15];
        double accurate = 0;

        int worstCount = 0;
        int epoch = 0;
        int count = 0;

        // Processing ANN
        while ((worstCount < stop) && (epoch < max)) {        // End if doing worst by the worstCount >= STOP or epoch is greater than max
            // Forward
            for (int k = 0; k < trainSize; k++) {
                // h0
                for (int i = 0; i < 15; i++) {
                    for (int j = 0; j < 22; j++) {
                        temp += trainX[k][j] * weight0[j][i];
                    }
                    h0[i] = sigmoid(temp);
                    temp = 0;
                }

                // h1
                for (int i = 0; i < 7; i++) {
                    for (int j = 0; j < 15; j++) {
                        temp += h0[j] * weight1[j][i];
                    }
                    h1[i] = sigmoid(temp);
                    temp = 0;
                }

//                // h1
//                for (int i = 0; i < 7; i++) {
//                    for (int j = 0; j < 22; j++) {
//                        temp += trainX[k][j] * weight1[j][i];
//                    }
//                    h1[i] = sigmoid(temp);
//                    temp = 0;
//                }

                // h2
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 7; j++) {
                        temp += h1[j] * weight2[j][i];
                    }
                    h2[i] = sigmoid(temp);
                    temp = 0;
                }

                // output
                for (int i = 0; i < 3; i++) {
                    temp += h2[i] * weight3[i][0];
                }
                output = sigmoid(temp);
                temp = 0;

                // Back Propagation
                double check;
                if (output < 0.5)
                    check = 0.0;
                else
                    check = 1.0;

                if (check == trainY[k]) {
                    count++;
                } else {                // Perform Back Propagation
                    // delta1 and weight3
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

//                    // delta3 and weight1
//                    double sum;
//                    for (int i = 0; i < 7; i++) {
//                        sum = 0;
//                        for (int j = 0; j < 3; j++) {
//                            sum += delta2[j] * weight2[i][j];
//                        }
//                        delta3[i] = h1[i] * (1 - h1[i]) * sum;
//                        for (int j = 0; j < 22; j++) {
//                            weight1[j][i] = weight1[j][i] + (learnRate * delta3[i] * h1[i]);
//                        }
//                    }

                    // delta3 and weight1
                    // delta3 and weight1
                    // delta3 and weight1
                    double sum;
                    for (int i = 0; i < 7; i++) {
                        sum = 0;
                        for (int j = 0; j < 3; j++) {
                            sum += delta2[j] * weight2[i][j];
                        }
                        delta3[i] = h1[i] * (1 - h1[i]) * sum;
                        for (int j = 0; j < 15; j++) {
                            weight1[j][i] = weight1[j][i] + (learnRate * delta3[i] * h1[i]);
                        }
                    }

                    // delta4 and weight0
                    for (int i = 0; i < 15; i++) {
                        sum = 0;
                        for (int j = 0; j < 7; j++) {
                            sum += delta3[j] * weight1[i][j];
                        }
                        delta4[i] = h0[i] * (1 - h0[i]) * sum;
                        for (int j = 0; j < 22; j++) {
                            weight0[j][i] = weight0[j][i] + (learnRate * delta4[i] * h0[i]);
                        }
                    }
                }
            }
            accurate = (double) count / (double) trainSize;
            count = 0; // reset count
            epoch++;

            // Look for stop point where it will increment +1 to worstCount if this Epoch is doing worst than the previous one
            if (accurate < prevAccurate) {
                worstCount++;
            }
            else {
                worstCount = 0;
            }
            prevAccurate = accurate;

            // Print Accuracy
            //System.out.println("Epoch " + epoch + " : " + accurate);

            if (epoch % 100 == 0) {
                System.out.println("Epoch " + epoch + " : " + accurate);
                if (epoch % 5000 == 0) {
                    learnRate = learnRate * 0.90;
                    System.out.println(learnRate);
                    printWeight(weight0, weight1, weight2, weight3);
                }
            }

        }
        return accurate;
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
     * @param rawData Original data
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
                weight[i][j] = -0.9 + (0.9+0.9) * random.nextDouble();
//                weight[i][j] = -2 + 4 * random.nextDouble();
//                weight[i][j] = -3 + 6 * random.nextDouble();

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
     * Print all model: Weight1, Weight2, Weight3
     * @param weight1 w1
     * @param weight2 w2
     * @param weight3 w3
     */
    private static void printWeight(double[][] weight0, double[][] weight1, double[][] weight2, double[][] weight3) {
        System.out.println("\n-----Weight 0-----");
        for (double[] aWeight0 : weight0) {
            for (double anAWeight0 : aWeight0) {
                System.out.format("%6.3f ", anAWeight0);
            }
            System.out.println();
        }

        System.out.println("\n-----Weight 1-----");
        for (double[] aWeight1 : weight1) {
            for (double anAWeight1 : aWeight1) {
                System.out.format("%6.3f ", anAWeight1);
            }
            System.out.println();
        }

        System.out.println("\n-----Weight 2-----");
        for (double[] aWeight2 : weight2) {
            for (double anAWeight2 : aWeight2) {
                System.out.format("%6.3f ", anAWeight2);
            }
            System.out.println();
        }

        System.out.println("\n-----Weight 3-----");
        for (double[] aWeight3 : weight3) {
            for (double anAWeight3 : aWeight3) {
                System.out.format("%6.3f ", anAWeight3);
            }
            System.out.println();
        }
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