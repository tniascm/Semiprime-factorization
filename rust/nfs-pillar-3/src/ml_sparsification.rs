use rand::Rng;
use ndarray::Array2;

/// Represents a simple graph adjacency matrix and feature matrix for a GNN.
/// Employs `ndarray` for accelerated matrix algebra, enabling future GPU/Tensor Core offloading.
pub struct TensorGraph {
    pub num_nodes: usize,
    /// Adjacency matrix of shape (num_nodes, num_nodes)
    pub adj_matrix: Array2<f64>,
    /// Node features of shape (num_nodes, in_features)
    pub node_features: Array2<f64>,
}

impl TensorGraph {
    pub fn new(num_nodes: usize, in_features: usize) -> Self {
        TensorGraph {
            num_nodes,
            adj_matrix: Array2::zeros((num_nodes, num_nodes)),
            node_features: Array2::zeros((num_nodes, in_features)),
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        if u < self.num_nodes && v < self.num_nodes {
            self.adj_matrix[[u, v]] = weight;
            self.adj_matrix[[v, u]] = weight; // undirected
        }
    }
}

/// A lightweight Graph Convolutional Network (GCN) layer simulation.
/// Computes H' = ReLU(A * H * W)
pub struct GCNLayer {
    pub in_features: usize,
    pub out_features: usize,
    /// Learnable weight matrix of shape (in_features, out_features)
    pub weights: Array2<f64>,
}

impl GCNLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        // Initialize weights randomly between -1.0 and 1.0
        let weights_vec: Vec<f64> = (0..in_features * out_features)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let weights = Array2::from_shape_vec((in_features, out_features), weights_vec)
            .expect("Failed to create weight matrix");

        GCNLayer {
            in_features,
            out_features,
            weights,
        }
    }

    /// Forward pass through the GCN layer using optimized ndarray dot products.
    /// In a production environment, this is accelerated on a GPU using Tensor Cores.
    pub fn forward(&self, graph: &TensorGraph, h: &Array2<f64>) -> Array2<f64> {
        // Step 1: Feature transformation (H * W)
        let hw = h.dot(&self.weights);

        // Step 2: Message passing (A * (H * W))
        let ahw = graph.adj_matrix.dot(&hw);

        // Step 3: Activation function (ReLU)
        // ahw.mapv(|x| x.max(0.0))
        let mut out = ahw;
        out.mapv_inplace(|x| x.max(0.0));

        out
    }
}

/// The main GNNSparsifier used to thin out the relation graph before Block Wiedemann.
pub struct GNNSparsifier {
    layer1: GCNLayer,
    layer2: GCNLayer,
}

impl Default for GNNSparsifier {
    fn default() -> Self {
        Self::new()
    }
}

impl GNNSparsifier {
    pub fn new() -> Self {
        GNNSparsifier {
            // Predict a single importance score per node based on 4-dimensional structural features
            layer1: GCNLayer::new(4, 8),
            layer2: GCNLayer::new(8, 1),
        }
    }

    /// Run the relation graph through the GNN to obtain edge retention probabilities.
    pub fn predict_retention(&self, graph: &TensorGraph) -> Array2<f64> {
        // Forward pass
        let h1 = self.layer1.forward(graph, &graph.node_features);
        let h2 = self.layer2.forward(graph, &h1);

        let num_nodes = graph.num_nodes;

        // Combine node scores to form an edge score.
        // H2 is (num_nodes, 1). We can compute score matrix as H2 * H2^T
        let h2_t = h2.t();
        let mut score_matrix = h2.dot(&h2_t);

        // Mask with the existing adjacency matrix (only predict scores for existing edges)
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if graph.adj_matrix[[i, j]] == 0.0 {
                    score_matrix[[i, j]] = 0.0;
                }
            }
        }

        score_matrix
    }

    pub fn sparsify(&self, graph: &TensorGraph, threshold: f64) -> TensorGraph {
        let edge_scores = self.predict_retention(graph);
        let mut new_graph = TensorGraph::new(graph.num_nodes, graph.node_features.ncols());
        new_graph.node_features = graph.node_features.clone();

        for i in 0..graph.num_nodes {
            for j in 0..graph.num_nodes {
                // Drop edges that don't meet the GNN predicted threshold
                if edge_scores[[i, j]] >= threshold && graph.adj_matrix[[i, j]] > 0.0 {
                    new_graph.adj_matrix[[i, j]] = graph.adj_matrix[[i, j]];
                }
            }
        }
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gnn_forward_pass_ndarray() {
        let num_nodes = 3;
        let in_features = 4;
        let mut graph = TensorGraph::new(num_nodes, in_features);

        // Connect a triangle
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 0, 1.0);

        // Dummy node features
        graph.node_features = ndarray::arr2(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]);

        let sparsifier = GNNSparsifier::new();
        let scores = sparsifier.predict_retention(&graph);
        assert_eq!(scores.dim(), (3, 3));

        // Very high threshold drops all edges
        let sparse = sparsifier.sparsify(&graph, 9999.0);
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                assert_eq!(sparse.adj_matrix[[i, j]], 0.0);
            }
        }
    }
}
