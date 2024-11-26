import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import numpy as np



def load_json_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def extract_weights_for_target(data, target_node):
    weights = []
    for edge in data["edges"]:
        if edge["target"] == target_node:
            weights.append(edge["metadata"]["weight"])
    return sum(weights)

def get_abcd_from_file(filename):
    data = load_json_from_file(filename)
    weights_for_A = extract_weights_for_target(data, "A")
    weights_for_B = extract_weights_for_target(data, "B")
    weights_for_C = extract_weights_for_target(data, "C")
    weights_for_D = extract_weights_for_target(data, "D")
    return [weights_for_A, weights_for_B, weights_for_C, weights_for_D]

def main():

    filename = 'data/duration_g10.json'  
    x = get_abcd_from_file(filename)

    filename = 'data/proxmity_g10.json'  
    y = get_abcd_from_file(filename)

    filename = 'data/attention_g10.json'
    z = get_abcd_from_file(filename)

    #data = list(zip(x, y))

    data = np.array(list(zip(x, y, z)))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for idx,i in enumerate(data):
        m = '^'
        col = '#ff7f0e'
        if kmeans.labels_[idx] == 1:
            m = 'o'
            col = '#e377c2'
        ax.scatter(i[0], i[1], i[2], marker=m, c=col)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()



if __name__ == "__main__":
    
    main()