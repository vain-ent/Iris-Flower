import ml.KNNClassifier;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        System.out.println("🌸 Iris Classifier 🌸");
        KNNClassifier knn = new KNNClassifier();
        knn.loadData("../data/iris.csv");

        Scanner sc = new Scanner(System.in);
        double[] input = new double[4];

        System.out.print("Enter Sepal Length: ");
        input[0] = sc.nextDouble();
        System.out.print("Enter Sepal Width: ");
        input[1] = sc.nextDouble();
        System.out.print("Enter Petal Length: ");
        input[2] = sc.nextDouble();
        System.out.print("Enter Petal Width: ");
        input[3] = sc.nextDouble();

        String species = knn.predict(input);
        System.out.println("Predicted Species: " + species);
        sc.close();
    }
}
