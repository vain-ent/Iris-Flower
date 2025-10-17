
import java.util.*;
import java.io.*;

public class KNNClassifier {
    private List<double[]> features = new ArrayList<>();
    private List<String> labels = new ArrayList<>();
    private int k = 3;

    // Load CSV dataset
    public void loadData(String path) {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split(",");
                if (p.length < 5) continue;
                double[] vals = {
                    Double.parseDouble(p[0]),
                    Double.parseDouble(p[1]),
                    Double.parseDouble(p[2]),
                    Double.parseDouble(p[3])
                };
                features.add(vals);
                labels.add(p[4]);
            }
        } catch (IOException e) {
            System.out.println("⚠️ Error loading data: " + e.getMessage());
        }
    }

    // Predict using KNN
    public String predict(double[] input) {
        List<double[]> distances = new ArrayList<>();
        for (int i = 0; i < features.size(); i++) {
            double dist = 0;
            for (int j = 0; j < input.length; j++)
                dist += Math.pow(input[j] - features.get(i)[j], 2);
            distances.add(new double[]{Math.sqrt(dist), i});
        }

        distances.sort(Comparator.comparingDouble(a -> a[0]));
        Map<String, Integer> count = new HashMap<>();

        for (int i = 0; i < k; i++) {
            String label = labels.get((int) distances.get(i)[1]);
            count.put(label, count.getOrDefault(label, 0) + 1);
        }

        return count.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .get().getKey();
    }
}
