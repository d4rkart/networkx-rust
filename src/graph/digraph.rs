use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug, Display, Formatter};
use rayon::prelude::*;

use super::graph::Graph;
use crate::NodeKey;

/// A directed graph implementation.
/// 
/// DiGraph stores nodes and edges with optional data/attributes.
/// DiGraphs hold directed edges. Self loops are allowed but multiple (parallel) edges are not.
/// 
/// # Type Parameters
/// 
/// * `T` - The type of data stored in nodes
/// * `E` - The type of data stored in edges (often called weights)
#[derive(Debug, Clone)]
pub struct DiGraph<T, E>
where
    T: Clone + Debug,
    E: Clone + Debug,
{
    // Inherit from Graph by composition
    graph: Graph<T, E>,
    // Additional predecessor map for directed edges
    pred: HashMap<NodeKey, HashMap<NodeKey, E>>,
}

impl<T, E> DiGraph<T, E>
where
    T: Clone + Debug,
    E: Clone + Debug,
{
    /// Creates a new empty directed graph.
    pub fn new() -> Self {
        DiGraph {
            graph: Graph::new(true),
            pred: HashMap::new(),
        }
    }

    /// Creates a new empty directed graph with a name.
    pub fn with_name(name: &str) -> Self {
        DiGraph {
            graph: Graph::with_name(true, name),
            pred: HashMap::new(),
        }
    }

    /// Returns the name of the graph.
    pub fn name(&self) -> &str {
        self.graph.name()
    }

    /// Sets the name of the graph.
    pub fn set_name(&mut self, name: &str) {
        self.graph.set_name(name)
    }

    /// Adds a node to the graph.
    pub fn add_node(&mut self, data: T) -> NodeKey {
        let key = self.graph.add_node(data);
        self.pred.insert(key, HashMap::new());
        key
    }

    /// Adds multiple nodes to the graph.
    pub fn add_nodes_from<I>(&mut self, nodes: I) -> Vec<NodeKey>
    where
        I: IntoIterator<Item = T>,
    {
        let keys = self.graph.add_nodes_from(nodes);
        for &key in &keys {
            self.pred.insert(key, HashMap::new());
        }
        keys
    }

    /// Gets the data associated with a node.
    pub fn get_node_data(&self, key: NodeKey) -> Option<&T> {
        self.graph.get_node_data(key)
    }

    /// Gets a mutable reference to the data associated with a node.
    pub fn get_node_data_mut(&mut self, key: NodeKey) -> Option<&mut T> {
        self.graph.get_node_data_mut(key)
    }

    /// Adds an edge between two nodes.
    pub fn add_edge(&mut self, from: NodeKey, to: NodeKey, weight: E) -> bool {
        if self.graph.add_edge(from, to, weight.clone()) {
            self.pred.entry(to).or_default().insert(from, weight);
            true
        } else {
            false
        }
    }

    /// Adds multiple edges to the graph.
    pub fn add_edges_from<I>(&mut self, edges: I) -> usize
    where
        I: IntoIterator<Item = (NodeKey, NodeKey, E)>,
    {
        let mut count = 0;
        for (from, to, weight) in edges {
            if self.add_edge(from, to, weight) {
                count += 1;
            }
        }
        count
    }

    /// Removes a node and all its incident edges from the graph.
    pub fn remove_node(&mut self, key: NodeKey) -> Option<T> {
        // First remove the node from the predecessor lists of all its successors
        if let Some(successors) = self.graph.adj().get(&key) {
            for &succ in successors.keys() {
                if let Some(pred_map) = self.pred.get_mut(&succ) {
                    pred_map.remove(&key);
                }
            }
        }

        // Then remove all predecessor relationships
        self.pred.remove(&key);

        // Remove from all predecessor lists where this node was a predecessor
        for pred_map in self.pred.values_mut() {
            pred_map.remove(&key);
        }

        // Finally remove the node from the underlying graph
        self.graph.remove_node(key)
    }

    /// Removes an edge from the graph.
    pub fn remove_edge(&mut self, from: NodeKey, to: NodeKey) -> Option<E> {
        if let Some(weight) = self.graph.remove_edge(from, to) {
            if let Some(pred_map) = self.pred.get_mut(&to) {
                pred_map.remove(&from);
            }
            Some(weight)
        } else {
            None
        }
    }

    /// Returns true if the graph has the specified edge.
    pub fn has_edge(&self, from: NodeKey, to: NodeKey) -> bool {
        self.graph.has_edge(from, to)
    }

    /// Returns true if node u has successor v.
    pub fn has_successor(&self, u: NodeKey, v: NodeKey) -> bool {
        self.graph.has_edge(u, v)
    }

    /// Returns true if node u has predecessor v.
    pub fn has_predecessor(&self, u: NodeKey, v: NodeKey) -> bool {
        self.pred.get(&u).map_or(false, |preds| preds.contains_key(&v))
    }

    /// Returns an iterator over successor nodes of n.
    pub fn successors(&self, n: NodeKey) -> impl Iterator<Item = NodeKey> + '_ {
        self.graph.neighbors(n).into_iter()
    }

    /// Returns an iterator over predecessor nodes of n.
    pub fn predecessors(&self, n: NodeKey) -> impl Iterator<Item = NodeKey> + '_ {
        self.pred
            .get(&n)
            .map(|preds| preds.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default()
            .into_iter()
    }

    /// Returns the number of predecessors of a node.
    pub fn in_degree(&self, n: NodeKey) -> usize {
        self.pred.get(&n).map_or(0, |preds| preds.len())
    }

    /// Returns the number of successors of a node.
    pub fn out_degree(&self, n: NodeKey) -> usize {
        self.graph.degree(Some(vec![n])).get(&n).copied().unwrap_or(0)
    }

    /// Returns the total degree (in + out) of a node.
    pub fn degree(&self, n: NodeKey) -> usize {
        self.in_degree(n) + self.out_degree(n)
    }

    /// Returns the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Creates a reversed version of the graph where all edges are reversed.
    pub fn reverse(&self) -> Self {
        let mut reversed = DiGraph::with_name(self.name());
        
        // Copy all nodes
        for key in self.graph.nodes() {
            if let Some(node_data) = self.get_node_data(key) {
                reversed.add_node(node_data.clone());
            }
        }

        // Add reversed edges using the adjacency map
        for (&from, neighbors) in self.graph.adj() {
            for (&to, weight) in neighbors {
                reversed.add_edge(to, from, weight.clone());
            }
        }

        reversed
    }

    /// Clears all nodes and edges from the graph.
    pub fn clear(&mut self) {
        self.graph.clear();
        self.pred.clear();
    }

    /// Removes all edges from the graph without altering nodes.
    pub fn clear_edges(&mut self) {
        self.graph.clear_edges();
        for pred_map in self.pred.values_mut() {
            pred_map.clear();
        }
    }

    /// Gets the weight of an edge between two nodes.
    pub fn get_edge_weight(&self, from: NodeKey, to: NodeKey) -> Option<&E> {
        self.graph.get_edge_weight(from, to)
    }

    /// Creates a new DiGraph containing the specified nodes and their edges.
    /// 
    /// # Arguments
    /// 
    /// * `nodes` - A vector of node keys to include in the subgraph
    /// 
    /// # Returns
    /// 
    /// A new DiGraph containing only the specified nodes and the edges between them.
    pub fn subgraph(&self, nodes: Vec<NodeKey>) -> Self {
        let mut subgraph = DiGraph::with_name(&format!("{}_subgraph", self.name()));
        
        // Add nodes with their original keys
        let node_set: std::collections::HashSet<_> = nodes.iter().cloned().collect();
        for &node in &nodes {
            if let Some(data) = self.get_node_data(node) {
                // Use the same key as in the original graph
                if subgraph.graph.add_node_with_key(node, data.clone()) {
                    subgraph.pred.insert(node, HashMap::new());
                }
            }
        }
        
        // Add edges that have both endpoints in the node set
        for (&from, neighbors) in self.graph.adj() {
            if node_set.contains(&from) {
                for (&to, weight) in neighbors {
                    if node_set.contains(&to) {
                        subgraph.add_edge(from, to, weight.clone());
                    }
                }
            }
        }
        
        subgraph
    }

    /// Returns a new DiGraph with all edges reversed (directions flipped), using parallel processing.
    ///
    /// # Returns
    ///
    /// A new DiGraph with the same nodes but reversed edges
    ///
    /// # Examples
    ///
    /// ```
    /// use networkx_rs::DiGraph;
    ///
    /// let mut graph = DiGraph::<i32, f64>::new();
    /// let n1 = graph.add_node(1);
    /// let n2 = graph.add_node(2);
    /// graph.add_edge(n1, n2, 1.0);
    ///
    /// let reversed = graph.reverse_par();
    /// assert!(reversed.has_edge(n2, n1));
    /// assert!(!reversed.has_edge(n1, n2));
    /// ```
    pub fn reverse_par(&self) -> Self
    where
        T: Send + Sync,
        E: Send + Sync,
    {
        let mut reversed = DiGraph::with_name(&self.name());
        
        // Add nodes in parallel
        let node_pairs: Vec<(NodeKey, T)> = self.graph.nodes()
            .par_iter()
            .filter_map(|&key| {
                self.graph.get_node_data(key).map(|data| (key, data.clone()))
            })
            .collect();
            
        // Add nodes sequentially to ensure keys match
        for (key, data) in node_pairs {
            let new_key = reversed.graph.add_node_with_key(key, data);
            debug_assert!(new_key);
        }
        
        // Collect all edges in parallel
        let reversed_edges: Vec<(NodeKey, NodeKey, E)> = self.graph.edges()
            .par_iter()
            .map(|(from, to, weight)| (*to, *from, weight.clone()))
            .collect();
            
        // Add reversed edges
        for (from, to, weight) in reversed_edges {
            reversed.graph.add_edge(from, to, weight);
        }
        
        reversed
    }

    /// Finds strongly connected components in the directed graph using Kosaraju's algorithm,
    /// with parallel processing where possible.
    ///
    /// # Returns
    ///
    /// A vector of components, where each component is a vector of node keys
    ///
    /// # Examples
    ///
    /// ```
    /// use networkx_rs::DiGraph;
    ///
    /// let mut graph = DiGraph::<&str, ()>::new();
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// let c = graph.add_node("C");
    ///
    /// graph.add_edge(a, b, ());
    /// graph.add_edge(b, c, ());
    /// graph.add_edge(c, a, ()); // Creates a cycle
    ///
    /// let components = graph.strongly_connected_components_par();
    /// assert_eq!(components.len(), 1); // One component containing all nodes
    /// assert_eq!(components[0].len(), 3);
    /// ```
    pub fn strongly_connected_components_par(&self) -> Vec<Vec<NodeKey>>
    where
        T: Send + Sync,
        E: Send + Sync,
    {
        let nodes = self.graph.nodes();
        if nodes.is_empty() {
            return Vec::new();
        }
        
        // First pass: DFS and collect finishing times
        let mut visited = HashSet::new();
        let mut finish_order = Vec::new();
        
        for &node in &nodes {
            if !visited.contains(&node) {
                self.dfs_finish_times(node, &mut visited, &mut finish_order);
            }
        }
        
        // Compute transposed graph in parallel
        let transposed = self.reverse_par();
        
        // Second pass: DFS on transposed graph in order of finishing times
        let mut visited = HashSet::new();
        let mut components = Vec::new();
        
        // We need to process the nodes in reverse finishing order
        for &node in finish_order.iter().rev() {
            if !visited.contains(&node) {
                let mut component = Vec::new();
                transposed.dfs_collect(node, &mut visited, &mut component);
                components.push(component);
            }
        }
        
        components
    }
    
    // Helper for strongly_connected_components_par
    fn dfs_finish_times(&self, node: NodeKey, visited: &mut HashSet<NodeKey>, finish_order: &mut Vec<NodeKey>) {
        visited.insert(node);
        
        if let Some(neighbors) = self.graph.adj().get(&node) {
            for &neighbor in neighbors.keys() {
                if !visited.contains(&neighbor) {
                    self.dfs_finish_times(neighbor, visited, finish_order);
                }
            }
        }
        
        finish_order.push(node);
    }
    
    // Helper for strongly_connected_components_par
    fn dfs_collect(&self, node: NodeKey, visited: &mut HashSet<NodeKey>, component: &mut Vec<NodeKey>) {
        visited.insert(node);
        component.push(node);
        
        if let Some(neighbors) = self.graph.adj().get(&node) {
            for &neighbor in neighbors.keys() {
                if !visited.contains(&neighbor) {
                    self.dfs_collect(neighbor, visited, component);
                }
            }
        }
    }

    /// Creates a subgraph from the specified nodes, using parallel execution.
    ///
    /// # Arguments
    ///
    /// * `nodes` - An iterator of node keys to include in the subgraph
    ///
    /// # Returns
    ///
    /// A new DiGraph containing only the specified nodes and edges between them
    ///
    /// # Examples
    ///
    /// ```
    /// use networkx_rs::DiGraph;
    ///
    /// let mut graph = DiGraph::<&str, i32>::new();
    /// let n1 = graph.add_node("A");
    /// let n2 = graph.add_node("B");
    /// let n3 = graph.add_node("C");
    ///
    /// graph.add_edge(n1, n2, 1);
    /// graph.add_edge(n2, n3, 2);
    ///
    /// let subgraph = graph.subgraph_par(vec![n1, n2]);
    /// assert_eq!(subgraph.node_count(), 2);
    /// assert_eq!(subgraph.edge_count(), 1);
    /// ```
    pub fn subgraph_par<I>(&self, nodes: I) -> Self
    where
        I: IntoIterator<Item = NodeKey>,
        T: Send + Sync,
        E: Send + Sync,
    {
        let mut subgraph = DiGraph::with_name(&self.name());
        let node_set: HashSet<NodeKey> = nodes.into_iter().collect();
        
        // Add nodes in parallel
        let node_pairs: Vec<(NodeKey, T)> = node_set.par_iter()
            .filter_map(|&key| {
                self.graph.get_node_data(key).map(|data| (key, data.clone()))
            })
            .collect();
            
        // Add nodes sequentially to ensure keys match
        for (key, data) in node_pairs {
            let new_key = subgraph.graph.add_node_with_key(key, data);
            debug_assert!(new_key);
        }
        
        // Collect all valid edges in parallel
        let valid_edges: Vec<(NodeKey, NodeKey, E)> = self.graph.edges()
            .par_iter()
            .filter_map(|(from, to, weight)| {
                if node_set.contains(from) && node_set.contains(to) {
                    Some((*from, *to, weight.clone()))
                } else {
                    None
                }
            })
            .collect();
            
        // Add edges
        for (from, to, weight) in valid_edges {
            subgraph.graph.add_edge(from, to, weight);
        }
        
        subgraph
    }
}

impl<T, E> Default for DiGraph<T, E>
where
    T: Clone + Debug,
    E: Clone + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, E> Display for DiGraph<T, E>
where
    T: Clone + Debug,
    E: Clone + Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DiGraph(name='{}', nodes={}, edges={})",
            self.name(),
            self.node_count(),
            self.edge_count()
        )
    }
}