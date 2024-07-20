import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

class DataUtils {
    public static List<Map<String, String>> loadCSV(String filePath) throws IOException {
        Reader reader = new FileReader(filePath);
        Iterable<CSVRecord> records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);
        List<Map<String, String>> data = new ArrayList<>();
        for (CSVRecord record : records) {
            Map<String, String> row = new HashMap<>();
            for (String header : record.toMap().keySet()) {
                row.put(header, record.get(header));
            }
            data.add(row);
        }
        return data;
    }
}

class DecisionTreeNode {
    String attribute;
    String value;
    DecisionTreeNode left;
    DecisionTreeNode right;
    boolean isLeaf;
    String label;

    DecisionTreeNode(String attribute) {
        this.attribute = attribute;
        this.isLeaf = false;
    }

    DecisionTreeNode(String attribute, boolean isLeaf, String label) {
        this.attribute = attribute;
        this.isLeaf = isLeaf;
        this.label = label;
    }
}

class DecisionTree {
    private DecisionTreeNode root;

    public void train(List<Map<String, String>> data, String targetAttribute) {
        root = buildTree(data, targetAttribute);
    }

    private DecisionTreeNode buildTree(List<Map<String, String>> data, String targetAttribute) {
        if (data.isEmpty()) {
            return new DecisionTreeNode(null, true, "unknown");
        }

        Set<String> uniqueLabels = new HashSet<>();
        for (Map<String, String> instance : data) {
            uniqueLabels.add(instance.get(targetAttribute));
        }

        if (uniqueLabels.size() == 1) {
            return new DecisionTreeNode(null, true, uniqueLabels.iterator().next());
        }

        String bestAttribute = chooseBestAttribute(data, targetAttribute);
        DecisionTreeNode node = new DecisionTreeNode(bestAttribute);

        Map<String, List<Map<String, String>>> partitions = partitionData(data, bestAttribute);
        node.left = buildTree(partitions.getOrDefault("yes", new ArrayList<>()), targetAttribute);
        node.right = buildTree(partitions.getOrDefault("no", new ArrayList<>()), targetAttribute);

        return node;
    }

    private String chooseBestAttribute(List<Map<String, String>> data, String targetAttribute) {
        String bestAttribute = null;
        double bestGain = 0;

        Set<String> attributes = data.get(0).keySet();
        attributes.remove(targetAttribute);

        for (String attribute : attributes) {
            double gain = calculateInformationGain(data, attribute, targetAttribute);
            if (gain > bestGain) {
                bestGain = gain;
                bestAttribute = attribute;
            }
        }
        return bestAttribute;
    }

    private double calculateInformationGain(List<Map<String, String>> data, String attribute, String targetAttribute) {
        double entropyBefore = calculateEntropy(data, targetAttribute);
        Map<String, List<Map<String, String>>> partitions = partitionData(data, attribute);

        double entropyAfter = 0;
        for (List<Map<String, String>> partition : partitions.values()) {
            double partitionWeight = (double) partition.size() / data.size();
            entropyAfter += partitionWeight * calculateEntropy(partition, targetAttribute);
        }
        return entropyBefore - entropyAfter;
    }

    private double calculateEntropy(List<Map<String, String>> data, String targetAttribute) {
        Map<String, Integer> labelCounts = new HashMap<>();
        for (Map<String, String> instance : data) {
            String label = instance.get(targetAttribute);
            labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
        }

        double entropy = 0;
        for (int count : labelCounts.values()) {
            double probability = (double) count / data.size();
            entropy -= probability * (Math.log(probability) / Math.log(2));
        }
        return entropy;
    }

    private Map<String, List<Map<String, String>>> partitionData(List<Map<String, String>> data, String attribute) {
        Map<String, List<Map<String, String>>> partitions = new HashMap<>();
        for (Map<String, String> instance : data) {
            String attributeValue = instance.get(attribute);
            partitions.computeIfAbsent(attributeValue, k -> new ArrayList<>()).add(instance);
        }
        return partitions;
    }

    public String classify(Map<String, String> instance) {
        DecisionTreeNode node = root;
        while (!node.isLeaf) {
            if (instance.get(node.attribute).equals("yes")) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        return node.label;
    }
}

public class DecisionTreeApp {
    public static void main(String[] args) {
        try {
            List<Map<String, String>> data = DataUtils.loadCSV("data.csv");
            DecisionTree tree = new DecisionTree();
            String targetAttribute = "target"; // add your target here
            tree.train(data, targetAttribute);

            int correct = 0;
            for (Map<String, String> instance : data) {
                String predicted = tree.classify(instance);
                if (predicted.equals(instance.get(targetAttribute))) {
                    correct++;
                }
            }
            double accuracy = (double) correct / data.size();
            System.out.println("Accuracy: " + accuracy);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}