import argparse
import os
import pickle as pk

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from utils import INTERMEDIATE_DATA_FOLDER_PATH, cosine_similarity_embeddings


def main(dataset_name, pca, cluster_method, lm_type, document_repr_type, random_state):
    # pca <= 0 means no pca
    do_pca = pca > 0

    save_dict_data = {"dataset_name": dataset_name,
                      "pca": pca,
                      "cluster_method": cluster_method,
                      "lm_type": lm_type,
                      "document_repr_type": document_repr_type,
                      "random_state": random_state}

    naming_suffix = f"pca{pca}.clus{cluster_method}.{lm_type}.{document_repr_type}.{random_state}"
    print(f"naming_suffix: {naming_suffix}")

    data_dir = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset_name)
    print(f"data_dir: {data_dir}")

    with open(os.path.join(data_dir, "dataset.pk"), "rb") as f:
        dictionary = pk.load(f)
        class_names = dictionary["class_names"]
        num_classes = len(class_names)
        print(class_names)

    with open(os.path.join(data_dir, f"document_repr_lm-{lm_type}-{document_repr_type}.pk"), "rb") as f:
        dictionary = pk.load(f)
        document_representations = dictionary["document_representations"]
        class_representations = dictionary["class_representations"]
        repr_prediction = np.argmax(cosine_similarity_embeddings(document_representations, class_representations),
                                    axis=1)
        save_dict_data["repr_prediction"] = repr_prediction

    if do_pca:
        print(f"Before fitting PCA. document_representations {document_representations.shape}; class_representations {class_representations.shape}")
        _pca = PCA(n_components=pca, random_state=random_state)
        document_representations = _pca.fit_transform(document_representations)
        class_representations = _pca.transform(class_representations)
        print(f"After fitting PCA. document_representations {document_representations.shape}; class_representations {class_representations.shape}")
        print(f"Explained variance: {sum(_pca.explained_variance_ratio_)}")

    if cluster_method == 'gmm':
        cosine_similarities = cosine_similarity_embeddings(document_representations, class_representations)
        document_class_assignment = np.argmax(cosine_similarities, axis=1)
        document_class_assignment_matrix = np.zeros((document_representations.shape[0], num_classes))
        for i in range(document_representations.shape[0]):
            document_class_assignment_matrix[i][document_class_assignment[i]] = 1.0

        gmm = GaussianMixture(n_components=num_classes, covariance_type='tied',
                              random_state=random_state,
                              n_init=999, warm_start=True)
        gmm.converged_ = "HACK"

        gmm._initialize(document_representations, document_class_assignment_matrix)
        gmm.lower_bound_ = -np.infty
        gmm.fit(document_representations)

        documents_to_class = gmm.predict(document_representations)
        centers = gmm.means_
        save_dict_data["centers"] = centers
        distance = -gmm.predict_proba(document_representations) + 1
    elif cluster_method == 'kmeans':
        kmeans = KMeans(n_clusters=num_classes, init=class_representations, random_state=random_state)
        kmeans.fit(document_representations)

        documents_to_class = kmeans.predict(document_representations)
        centers = kmeans.cluster_centers_
        save_dict_data["centers"] = centers
        distance = np.zeros((document_representations.shape[0], centers.shape[0]), dtype=float)
        for i, _emb_a in enumerate(document_representations):
            for j, _emb_b in enumerate(centers):
                distance[i][j] = np.linalg.norm(_emb_a - _emb_b)

    save_dict_data["documents_to_class"] = documents_to_class
    save_dict_data["distance"] = distance

    with open(os.path.join(data_dir, f"data.{naming_suffix}.pk"), "wb") as f:
        print(f"Saving class-aligned documents to {data_dir}/data.{naming_suffix}.pk")
        pk.dump(save_dict_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="nyt_topics")
    parser.add_argument("--pca", type=int, default=64, help="number of dimensions projected to in PCA, "
                                                            "-1 means not doing PCA.")
    parser.add_argument("--cluster_method", choices=["gmm", "kmeans"], default="gmm")
    # language model + layer
    parser.add_argument("--lm_type", default="bbu-12")
    # attention mechanism + T
    parser.add_argument("--document_repr_type", default="mixture-100")
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    print(vars(args))
    main(args.dataset_name, args.pca, args.cluster_method, args.lm_type, args.document_repr_type, args.random_state)
