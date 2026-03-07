use rand::Rng;

/// Represents a simple graph adjacency matrix and feature matrix for a GNN.
/// In a full system, `ndarray` or `tch` would be used for matrix algebra.
pub struct TensorGraph {
    pub num_nodes: usize,
    /// Adjacency matrix flattened
    pub adj_matrix: Vec<f64>,
    /// Node features flattened (size: num_nodes * in_features)
    pub node_features: Vec<f64>,
}

impl TensorGraph {
    pub fn new(num_nodes: usize, in_features: usize) -> Self {
        TensorGraph {
            num_nodes,
            adj_matrix: vec![0.0; num_nodes * num_nodes],
            node_features: vec![0.0; num_nodes * in_features],
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        if u < self.num_nodes && v < self.num_nodes {
            self.adj_matrix[u * self.num_nodes + v] = weight;
            self.adj_matrix[v * self.num_nodes + u] = weight; // undirected
        }
    }
}

/// A lightweight Graph Convolutional Network (GCN) layer simulation.
/// Computes H' = ReLU(A * H * W)
pub struct GCNLayer {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Vec<f64>,
}

impl GCNLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        // Initialize weights randomly
        let weights = (0..in_features * out_features).map(|_| rng.gen_range(-1.0..1.0)).collect();
        GCNLayer {
            in_features,
            out_features,
            weights,
        }
    }

    /// Forward pass through the GCN layer.
    /// In a production environment, this is accelerated on a GPU using Tensor Cores.
    pub fn forward(&self, graph: &TensorGraph, h: &[f64]) -> Vec<f64> {
        let num_nodes = graph.num_nodes;
        // Step 1: Feature transformation (H * W)
        let mut h_w = vec![0.0; num_nodes * self.out_features];
        for i in 0..num_nodes {
            for j in 0..self.out_features {
                let mut sum = 0.0;
                for k in 0..self.in_features {
                    sum += h[i * self.in_features + k] * self.weights[k * self.out_features + j];
                }
                h_w[i * self.out_features + j] = sum;
            }
        }

        // Step 2: Message passing (A * (H * W))
        let mut out = vec![0.0; num_nodes * self.out_features];
        for i in 0..num_nodes {
            for j in 0..self.out_features {
                let mut sum = 0.0;
                for k in 0..num_nodes {
                    // Adjacency matrix element A_{ik}
                    let a_ik = graph.adj_matrix[i * num_nodes + k];
                    sum += a_ik * h_w[k * self.out_features + j];
                }
                // Step 3: Activation function (ReLU)
                out[i * self.out_features + j] = sum.max(0.0);
            }
        }

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
    pub fn predict_retention(&self, graph: &TensorGraph) -> Vec<f64> {
        // Forward pass
        let h1 = self.layer1.forward(graph, &graph.node_features);
        let h2 = self.layer2.forward(graph, &h1);

        let num_nodes = graph.num_nodes;
        let mut edge_scores = vec![0.0; num_nodes * num_nodes];

        // Combine node scores to form an edge score.
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if graph.adj_matrix[i * num_nodes + j] > 0.0 {
                    // Predict edge weight by taking the product (or sum) of node embeddings
                    let score = h2[i * self.layer2.out_features] * h2[j * self.layer2.out_features];
                    edge_scores[i * num_nodes + j] = score;
                }
            }
        }
        edge_scores
    }

    pub fn sparsify(&self, graph: &TensorGraph, threshold: f64) -> TensorGraph {
        let edge_scores = self.predict_retention(graph);
        let mut new_graph = TensorGraph::new(graph.num_nodes, graph.node_features.len() / graph.num_nodes);
        new_graph.node_features = graph.node_features.clone();

        for i in 0..graph.num_nodes {
            for j in 0..graph.num_nodes {
                // Drop edges that don't meet the GNN predicted threshold
                if edge_scores[i * graph.num_nodes + j] >= threshold {
                    new_graph.adj_matrix[i * graph.num_nodes + j] = graph.adj_matrix[i * graph.num_nodes + j];
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
    fn test_gnn_forward_pass() {
        let mut graph = TensorGraph::new(3, 4);
        // Connect a triangle
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 0, 1.0);

        // Dummy node features
        graph.node_features = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];

        let sparsifier = GNNSparsifier::new();
        let scores = sparsifier.predict_retention(&graph);
        assert_eq!(scores.len(), 9);

        // Very high threshold drops all edges
        let sparse = sparsifier.sparsify(&graph, 9999.0);
        for &weight in &sparse.adj_matrix {
            assert_eq!(weight, 0.0);
        }
    }
}
