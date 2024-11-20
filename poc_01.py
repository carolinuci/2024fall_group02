import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json



def load_json_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def extract_weights_for_target(data, target_node):
    weights = []
    for edge in data["edges"]:
        if edge["target"] == target_node:
            weights.append(edge["metadata"]["weight"])
    return sum(weights)


def main():

    inertias = []
    filename = 'data/duration_g10.json'  
    data = load_json_from_file(filename)
    weights_for_A = extract_weights_for_target(data, "A")
    weights_for_B = extract_weights_for_target(data, "B")
    weights_for_C = extract_weights_for_target(data, "C")
    weights_for_D = extract_weights_for_target(data, "D")
    x = [weights_for_A, weights_for_B, weights_for_C, weights_for_D]
    

    filename_prox = 'data/proxmity_g10.json'  
    data = load_json_from_file(filename_prox)
    weights_for_A = extract_weights_for_target(data, "A")
    weights_for_B = extract_weights_for_target(data, "B")
    weights_for_C = extract_weights_for_target(data, "C")
    weights_for_D = extract_weights_for_target(data, "D")
    y = [weights_for_A, weights_for_B, weights_for_C, weights_for_D]
    data = list(zip(x, y))


    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)

    plt.scatter(x, y, c=kmeans.labels_)
    plt.show()


if __name__ == "__main__":
    
    main()