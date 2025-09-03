use std::collections::{HashMap, HashSet};
use std::fmt;

use super::graph::NodeKey;
use super::multigraph::MultiGraph;

/// A directed multigraph implementation that allows multiple edges between the same nodes.
///
/// MultiDiGraph stores nodes and edges with optional data/attributes.
/// MultiDiGraphs hold directed edges. Self loops are allowed and multiple (parallel) edges are allowed.
///
/// # Type Parameters
///
/// * `T` - The type of data stored in nodes
/// * `E` - The type of data stored in edges (often called weights)
///
/// # Examples
///
/// ```
/// use networkx_rs::MultiDiGraph;
///
/// let mut graph = MultiDiGraph::<String, f64>::new();
///
/// // Add nodes
/// let n1 = graph.add_node("A".to_string());
/// let n2 = graph.add_node("B".to_string());
///
/// // Add multiple edges between the same nodes
/// let key1 = graph.add_edge(n1, n2, 1.0);
/// let key2 = graph.add_edge(n1, n2, 2.0);
///
/// assert_eq!(graph.number_of_edges(n1, n2), 2);
/// ```
#[derive(Debug, Clone)]
pub struct MultiDiGraph<T, E>
where
    T: Clone + std::fmt::Debug,
    E: Clone + std::fmt::Debug,
{
    // Use MultiGraph by composition
    multigraph: MultiGraph<T, E>,
    // Additional predecessor mapping for directed edges
    pred: HashMap<NodeKey, HashMap<NodeKey, HashMap<String, E>>>,
}

impl<T, E> MultiDiGraph<T, E>
where
    T: Clone + std::fmt::Debug + Default,
    E: Clone + std::fmt::Debug,
{
    /// Creates a new empty multi-directed graph.
    pub fn new() -> Self {
        MultiDiGraph {
            multigraph: MultiGraph::new(),
            pred: HashMap::new(),
        }
    }

    // /// Creates a new empty multi-directed graph with a name.
    // pub fn with_name(_name: &str) -> Self {
    //     // UNUSED
    //     // Return a basic MultiDiGraph
    //     // The name functionality could be added when MultiGraph has a public
    //     // graph method or name setter
    //     MultiDiGraph::new()
    // }

    // /// Returns the name of the graph.
    // pub fn name(&self) -> &str {
    //     // Default to empty string since we can't access the graph's name directly
    //     ""
    // }

    // /// Sets the name of the graph.
    // pub fn set_name(&mut self, _name: &str) {
    //     // Can't directly access the graph to set the name
    // }

    /// Adds a node to the graph.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to associate with the node
    ///
    /// # Returns
    ///
    /// The key of the newly added node
    pub fn add_node(&mut self, data: T) -> NodeKey {
        let key = self.multigraph.add_node(data);
        self.pred.insert(key, HashMap::new());
        key
    }

    /// Adds multiple nodes to the graph.
    pub fn add_nodes_from<I>(&mut self, nodes: I) -> Vec<NodeKey>
    where
        I: IntoIterator<Item = T>,
    {
        let mut keys = Vec::new();
        for node in nodes {
            keys.push(self.add_node(node));
        }
        keys
    }

    /// Adds an edge between two nodes with a given weight.
    /// Returns the edge key.
    ///
    /// # Arguments
    ///
    /// * `from` - The key of the source node
    /// * `to` - The key of the target node
    /// * `weight` - The weight to associate with the edge
    ///
    /// # Returns
    ///
    /// A unique string key that identifies this edge
    pub fn add_edge(&mut self, from: NodeKey, to: NodeKey, weight: E) -> String {
        // Create nodes if they don't exist
        if !self.multigraph.has_node(from) {
            self.add_node(T::default());
        }
        if !self.multigraph.has_node(to) {
            self.add_node(T::default());
        }

        // Add the edge to the forward adjacency map
        let key = self.multigraph.add_edge(from, to, weight.clone());
        
        // Add the edge to the predecessor map
        self.pred.entry(to).or_default().entry(from).or_default().insert(key.clone(), weight);
        
        key
    }

    /// Adds an edge with a specified key.
    pub fn add_edge_with_key(&mut self, from: NodeKey, to: NodeKey, key: String, weight: E) -> String {
        // Create nodes if they don't exist
        if !self.multigraph.has_node(from) {
            self.add_node(T::default());
        }
        if !self.multigraph.has_node(to) {
            self.add_node(T::default());
        }

        // Add the edge to the forward adjacency map
        let edge_key = self.multigraph.add_edge_with_key(from, to, key.clone(), weight.clone());
        
        // Add the edge to the predecessor map
        self.pred.entry(to).or_default().entry(from).or_default().insert(edge_key.clone(), weight);
        
        edge_key
    }

    /// Adds multiple edges to the graph.
    pub fn add_edges_from<I>(&mut self, edges: I) -> Vec<String>
    where
        I: IntoIterator<Item = (NodeKey, NodeKey, E)>,
    {
        let mut keys = Vec::new();
        for (from, to, weight) in edges {
            keys.push(self.add_edge(from, to, weight));
        }
        keys
    }

    /// Removes a node and all its incident edges from the graph.
    ///
    /// # Arguments
    ///
    /// * `node` - The key of the node to remove
    ///
    /// # Returns
    ///
    /// An option containing the data of the removed node, or None if the node doesn't exist
    pub fn remove_node(&mut self, node: NodeKey) -> Option<T> {
        // Remove all edges where this node is a target
        if let Some(predecessors) = self.pred.get(&node).cloned() {
            for (pred, edge_dict) in predecessors {
                for (key, _) in edge_dict {
                    self.multigraph.remove_edge(pred, node, &key);
                }
            }
        }
        
        // Remove all edges where this node is a source
        let neighbors = self.multigraph.neighbors(node);
        for neighbor in neighbors {
            if let Some(edges_between) = self.edges_between_as_map(node, neighbor) {
                let keys: Vec<String> = edges_between.keys().cloned().collect();
                for key in keys {
                    if let Some(pred_dict) = self.pred.get_mut(&neighbor) {
                        if let Some(edge_dict) = pred_dict.get_mut(&node) {
                            edge_dict.remove(&key);
                        }
                        if pred_dict.get(&node).map_or(true, |d| d.is_empty()) {
                            pred_dict.remove(&node);
                        }
                    }
                }
            }
        }
        
        // Remove from predecessor mapping
        self.pred.remove(&node);
        
        // Finally remove the node from the underlying graph
        self.multigraph.remove_node(node)
    }

    // Helper method to get edges between nodes as a map
    fn edges_between_as_map(&self, from: NodeKey, to: NodeKey) -> Option<HashMap<String, E>> {
        let edges_between = self.multigraph.edges_between(from, to);
        if edges_between.is_empty() {
            None
        } else {
            let mut result = HashMap::new();
            for (key, weight) in edges_between {
                result.insert(key, weight.clone());
            }
            Some(result)
        }
    }

    /// Removes an edge between two nodes.
    /// Returns the weight of the removed edge.
    ///
    /// # Arguments
    ///
    /// * `from` - The key of the source node
    /// * `to` - The key of the target node
    /// * `key` - The unique key of the edge to remove
    ///
    /// # Returns
    ///
    /// An option containing the weight of the removed edge, or None if the edge doesn't exist
    pub fn remove_edge(&mut self, from: NodeKey, to: NodeKey, key: &str) -> Option<E> {
        if let Some(pred_dict) = self.pred.get_mut(&to) {
            if let Some(edge_dict) = pred_dict.get_mut(&from) {
                edge_dict.remove(key);
                if edge_dict.is_empty() {
                    pred_dict.remove(&from);
                }
            }
        }
        
        // Then remove the edge from the multigraph
        self.multigraph.remove_edge(from, to, key)
    }

    /// Returns true if the graph contains the given node.
    ///
    /// # Arguments
    ///
    /// * `node` - The key of the node to check
    ///
    /// # Returns
    ///
    /// True if the node exists, false otherwise
    pub fn has_node(&self, node: NodeKey) -> bool {
        self.multigraph.has_node(node)
    }

    /// Returns true if the graph contains an edge between the given nodes with the given key.
    ///
    /// # Arguments
    ///
    /// * `from` - The key of the source node
    /// * `to` - The key of the target node
    /// * `key` - The unique key of the edge to check
    ///
    /// # Returns
    ///
    /// True if the edge exists, false otherwise
    pub fn has_edge(&self, from: NodeKey, to: NodeKey, key: &str) -> bool {
        self.multigraph.has_edge(from, to, key)
    }

    /// Returns true if node u has successor v.
    pub fn has_successor(&self, u: NodeKey, v: NodeKey) -> bool {
        self.number_of_edges(u, v) > 0
    }

    /// Returns true if node u has predecessor v.
    pub fn has_predecessor(&self, u: NodeKey, v: NodeKey) -> bool {
        if let Some(pred_dict) = self.pred.get(&u) {
            pred_dict.contains_key(&v)
        } else {
            false
        }
    }

    /// Returns the weight of an edge between two nodes.
    ///
    /// # Arguments
    ///
    /// * `from` - The key of the source node
    /// * `to` - The key of the target node
    /// * `key` - The unique key of the edge
    ///
    /// # Returns
    ///
    /// An option containing a reference to the edge weight, or None if the edge doesn't exist
    pub fn get_edge_weight(&self, from: NodeKey, to: NodeKey, key: &str) -> Option<&E> {
        self.multigraph.get_edge_weight(from, to, key)
    }

    /// Returns all successors of a node.
    ///
    /// # Arguments
    ///
    /// * `node` - The key of the node to get successors for
    ///
    /// # Returns
    ///
    /// A HashSet of node keys that are successors of the given node
    pub fn successors(&self, node: NodeKey) -> HashSet<NodeKey> {
        self.multigraph.neighbors(node)
    }

    /// Returns all predecessors of a node.
    ///
    /// # Arguments
    ///
    /// * `node` - The key of the node to get predecessors for
    ///
    /// # Returns
    ///
    /// A HashSet of node keys that are predecessors of the given node
    pub fn predecessors(&self, node: NodeKey) -> HashSet<NodeKey> {
        if let Some(pred_dict) = self.pred.get(&node) {
            pred_dict.keys().cloned().collect()
        } else {
            HashSet::new()
        }
    }

    /// Returns all edges in the graph.
    ///
    /// # Returns
    ///
    /// A vector of (source, target, key, weight) tuples representing edges
    pub fn edges(&self) -> Vec<(NodeKey, NodeKey, String, E)> {
        let mut result = Vec::new();
        
        for from in self.nodes() {
            let successors = self.successors(from);
            for to in successors {
                for (key, weight) in self.edges_between(from, to) {
                    result.push((from, to, key.clone(), weight.clone()));
                }
            }
        }
        
        result
    }

    /// Returns all edges in the graph using parallel processing.
    ///
    /// # Returns
    ///
    /// A vector of (source, target, key, weight) tuples representing edges
    pub fn edges_par(&self) -> Vec<(NodeKey, NodeKey, String, E)>
    where
        T: Send + Sync,
        E: Clone + Send + Sync,
    {
        // TODO: Implement parallel edge iteration
        self.edges()
    }

    /// Returns the number of edges between two nodes.
    ///
    /// # Arguments
    ///
    /// * `from` - The key of the source node
    /// * `to` - The key of the target node
    ///
    /// # Returns
    ///
    /// The number of edges between the two nodes
    pub fn number_of_edges(&self, from: NodeKey, to: NodeKey) -> usize {
        if !self.has_node(from) || !self.has_node(to) {
            return 0;
        }

        let edges = self.edges_between(from, to);
        edges.len()
    }

    /// Returns all edges between two nodes.
    ///
    /// # Arguments
    ///
    /// * `from` - The key of the source node
    /// * `to` - The key of the target node
    ///
    /// # Returns
    ///
    /// A vector of (key, weight) tuples representing edges between the two nodes
    pub fn edges_between(&self, from: NodeKey, to: NodeKey) -> Vec<(String, &E)> {
        if !self.has_node(from) || !self.has_node(to) {
            return Vec::new();
        }
        
        // Use the predecessor map to verify the edge direction
        if let Some(pred_dict) = self.pred.get(&to) {
            if let Some(edge_dict) = pred_dict.get(&from) {
                // Now we know we have a directed edge from->to
                // We still need to get the proper references from MultiGraph
                let edges_from_multigraph = self.multigraph.edges_between(from, to);
                
                // Filter to only include the keys that exist in the predecessor map
                return edges_from_multigraph.into_iter()
                    .filter(|(key, _)| edge_dict.contains_key(key))
                    .collect();
            }
        }
        
        Vec::new()
    }

    /// Returns the in-degree of a node.
    ///
    /// # Arguments
    ///
    /// * `node` - The key of the node
    ///
    /// # Returns
    ///
    /// The number of edges pointing to the node
    pub fn in_degree(&self, node: NodeKey) -> usize {
        if let Some(pred_dict) = self.pred.get(&node) {
            pred_dict.values().map(|edges| edges.len()).sum()
        } else {
            0
        }
    }

    /// Returns the out-degree of a node.
    ///
    /// # Arguments
    ///
    /// * `node` - The key of the node
    ///
    /// # Returns
    ///
    /// The number of edges pointing from the node
    pub fn out_degree(&self, node: NodeKey) -> usize {
        let neighbors = self.multigraph.neighbors(node);
        neighbors.iter().map(|&neighbor| self.number_of_edges(node, neighbor)).sum()
    }

    /// Returns the total degree (in + out) of a node.
    ///
    /// # Arguments
    ///
    /// * `node` - The key of the node
    ///
    /// # Returns
    ///
    /// The total number of edges connected to the node
    pub fn degree(&self, node: NodeKey) -> usize {
        self.in_degree(node) + self.out_degree(node)
    }

    /// Returns the number of nodes in the graph.
    ///
    /// # Returns
    ///
    /// The number of nodes
    pub fn node_count(&self) -> usize {
        self.multigraph.node_count()
    }

    /// Returns the number of edges in the graph.
    ///
    /// # Returns
    ///
    /// The number of edges
    pub fn edge_count(&self) -> usize {
        // Count edges from our edges method rather than relying on multigraph
        self.edges().len()
    }

    /// Returns true as this is a multigraph.
    ///
    /// # Returns
    ///
    /// Always true
    pub fn is_multigraph(&self) -> bool {
        true
    }

    /// Returns true as this is a directed graph.
    ///
    /// # Returns
    ///
    /// Always true
    pub fn is_directed(&self) -> bool {
        true
    }

    /// Clears all nodes and edges from the graph.
    pub fn clear(&mut self) {
        self.multigraph.clear();
        self.pred.clear();
    }

    /// Returns a new graph that is a subgraph of this graph induced by the specified nodes.
    ///
    /// # Arguments
    ///
    /// * `nodes` - A vector of node keys to include in the subgraph
    ///
    /// # Returns
    ///
    /// A new MultiDiGraph containing only the specified nodes and their incident edges
    pub fn subgraph(&self, nodes: Vec<NodeKey>) -> MultiDiGraph<T, E> {
        let mut subgraph = MultiDiGraph::new();
        
        // Add the nodes
        for &node in &nodes {
            if let Some(data) = self.multigraph.get_node_data(node) {
                let new_node = subgraph.add_node(data.clone());
                assert_eq!(node, new_node);
            }
        }
        
        // Add the edges
        let nodes_set: HashSet<NodeKey> = nodes.into_iter().collect();
        for (u, v, key, weight) in self.edges() {
            if nodes_set.contains(&u) && nodes_set.contains(&v) {
                subgraph.add_edge_with_key(u, v, key, weight);
            }
        }
        
        subgraph
    }

    /// Returns all nodes in the graph.
    ///
    /// # Returns
    ///
    /// A vector of node keys
    pub fn nodes(&self) -> Vec<NodeKey> {
        self.multigraph.nodes()
    }

    /// Gets the data associated with a node.
    ///
    /// # Arguments
    ///
    /// * `node` - The key of the node
    ///
    /// # Returns
    ///
    /// An option containing a reference to the node data, or None if the node doesn't exist
    pub fn get_node_data(&self, node: NodeKey) -> Option<&T> {
        self.multigraph.get_node_data(node)
    }

    /// Access the underlying multigraph for more operations.
    pub fn multigraph(&self) -> &MultiGraph<T, E> {
        &self.multigraph
    }

    /// Access the underlying multigraph for mutable operations.
    pub fn multigraph_mut(&mut self) -> &mut MultiGraph<T, E> {
        &mut self.multigraph
    }

    /// Creates a reversed copy of the graph.
    ///
    /// # Returns
    ///
    /// A new MultiDiGraph with all edges reversed
    pub fn reverse(&self) -> Self {
        let mut reversed = MultiDiGraph::new();
        
        // Copy all nodes
        for node in self.nodes() {
            if let Some(data) = self.get_node_data(node) {
                let _new_node = reversed.add_node(data.clone());
            }
        }
        
        // Create the reversed graph manually
        // In a directed graph, we need to reverse the direction of each edge
        for (from, to, key, weight) in self.edges() {
            // Add the edge in the reverse direction
            reversed.add_edge_with_key(to, from, key, weight); 
        }
        
        reversed
    }

    /// Converts the graph to an undirected multigraph.
    ///
    /// # Returns
    ///
    /// A new MultiGraph with the same nodes and edges but without direction
    pub fn to_undirected(&self) -> MultiGraph<T, E> {
        let mut undirected = MultiGraph::new();
        
        // Copy all nodes
        for node in self.nodes() {
            if let Some(data) = self.get_node_data(node) {
                let _new_node = undirected.add_node(data.clone());
                // Don't assert equality of node IDs, as they might change
                // Just ensure all nodes are copied
            }
        }
        
        // Copy all edges, preserving their keys
        for (u, v, key, weight) in self.edges() {
            undirected.add_edge_with_key(u, v, key, weight);
        }
        
        undirected
    }
}

impl<T, E> Default for MultiDiGraph<T, E>
where
    T: Clone + std::fmt::Debug + Default,
    E: Clone + std::fmt::Debug,
{
    fn default() -> Self {
        MultiDiGraph::new()
    }
}

impl<T, E> fmt::Display for MultiDiGraph<T, E>
where
    T: Clone + std::fmt::Debug + Default,
    E: Clone + std::fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MultiDiGraph with {} nodes and {} edges", self.node_count(), self.edge_count())
    }
}

impl<T, E> PartialEq for MultiDiGraph<T, E>
where
    T: Clone + std::fmt::Debug + PartialEq + Default,
    E: Clone + std::fmt::Debug + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        // Check if both graphs have the same nodes
        if self.node_count() != other.node_count() {
            return false;
        }
        
        // Check if both graphs have the same edges
        if self.edge_count() != other.edge_count() {
            return false;
        }
        
        // Check if all nodes in self are in other with the same data
        for node in self.nodes() {
            if let Some(self_data) = self.get_node_data(node) {
                if let Some(other_data) = other.get_node_data(node) {
                    if self_data != other_data {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
        
        // Check if all edges in self are in other with the same data
        for (u, v, key, weight) in self.edges() {
            if let Some(other_weight) = other.get_edge_weight(u, v, &key) {
                if &weight != other_weight {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        true
    }
} 