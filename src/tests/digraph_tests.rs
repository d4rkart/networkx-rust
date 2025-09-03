use crate::DiGraph;

#[cfg(test)]
mod digraph_tests {
    use super::*;
    
    // Basic functionality tests
    
    #[test]
    fn test_new_digraph() {
        let graph = DiGraph::<String, i32>::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }
    
    #[test]
    fn test_digraph_with_name() {
        let graph = DiGraph::<i32, f64>::with_name("Named DiGraph");
        assert_eq!(graph.name(), "Named DiGraph");
        assert_eq!(graph.node_count(), 0);
    }
    
    #[test]
    fn test_basic_node_operations() {
        let mut graph = DiGraph::<String, i32>::new();
        
        // Add nodes
        let n1 = graph.add_node("Node 1".to_string());
        let n2 = graph.add_node("Node 2".to_string());
        
        // Check node count
        assert_eq!(graph.node_count(), 2);
        
        // Check node existence
        assert!(graph.get_node_data(n1).is_some());
        assert!(graph.get_node_data(n2).is_some());
        assert!(graph.get_node_data(999).is_none()); // Non-existent node
        
        // Check node data
        assert_eq!(graph.get_node_data(n1), Some(&"Node 1".to_string()));
        assert_eq!(graph.get_node_data(n2), Some(&"Node 2".to_string()));
        
        // Check node data mutation
        if let Some(data) = graph.get_node_data_mut(n1) {
            *data = "Updated Node 1".to_string();
        }
        assert_eq!(graph.get_node_data(n1), Some(&"Updated Node 1".to_string()));
        
        // Remove node
        assert_eq!(graph.remove_node(n1), Some("Updated Node 1".to_string()));
        assert_eq!(graph.node_count(), 1);
        assert!(graph.get_node_data(n1).is_none());
        assert!(graph.get_node_data(n2).is_some());
    }
    
    #[test]
    fn test_edge_operations() {
        let mut graph = DiGraph::<i32, String>::new();
        
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        
        // Add edges
        assert!(graph.add_edge(n1, n2, "Edge 1->2".to_string()));
        assert!(graph.add_edge(n2, n3, "Edge 2->3".to_string()));
        
        // Check edge count
        assert_eq!(graph.edge_count(), 2);
        
        // Check edge existence
        assert!(graph.has_edge(n1, n2));
        assert!(graph.has_edge(n2, n3));
        assert!(!graph.has_edge(n1, n3)); // Non-existent edge
        assert!(!graph.has_edge(n2, n1)); // Opposite direction
        
        // Check edge data
        assert_eq!(graph.get_edge_weight(n1, n2), Some(&"Edge 1->2".to_string()));
        assert_eq!(graph.get_edge_weight(n2, n3), Some(&"Edge 2->3".to_string()));
        
        // Remove and add edge with new weight (simulates an update)
        graph.remove_edge(n1, n2);
        assert!(graph.add_edge(n1, n2, "Updated Edge 1->2".to_string()));
        assert_eq!(graph.get_edge_weight(n1, n2), Some(&"Updated Edge 1->2".to_string()));
        
        // Remove edge
        assert_eq!(graph.remove_edge(n1, n2), Some("Updated Edge 1->2".to_string()));
        assert_eq!(graph.edge_count(), 1);
        assert!(!graph.has_edge(n1, n2));
        assert!(graph.has_edge(n2, n3));
    }
    
    #[test]
    fn test_node_removal_cascade() {
        let mut graph = DiGraph::<char, i32>::new();
        
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        
        graph.add_edge(a, b, 1);
        graph.add_edge(b, c, 2);
        graph.add_edge(c, a, 3);
        
        assert_eq!(graph.edge_count(), 3);
        
        // Remove node and check that its edges are also removed
        graph.remove_node(b);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1); // Only edge c->a should remain
        assert!(graph.has_edge(c, a));
        assert!(!graph.has_edge(a, b));
        assert!(!graph.has_edge(b, c));
    }
    
    #[test]
    fn test_in_out_degree() {
        let mut graph = DiGraph::<i32, ()>::new();
        
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        
        graph.add_edge(n1, n2, ());
        graph.add_edge(n1, n3, ());
        graph.add_edge(n2, n3, ());
        
        // Check in-degree
        assert_eq!(graph.in_degree(n1), 0);
        assert_eq!(graph.in_degree(n2), 1);
        assert_eq!(graph.in_degree(n3), 2);
        
        // Check out-degree
        assert_eq!(graph.out_degree(n1), 2);
        assert_eq!(graph.out_degree(n2), 1);
        assert_eq!(graph.out_degree(n3), 0);
        
        // Check total degree
        assert_eq!(graph.degree(n1), 2);
        assert_eq!(graph.degree(n2), 2);
        assert_eq!(graph.degree(n3), 2);
    }
    
    #[test]
    fn test_predecessors_successors() {
        let mut graph = DiGraph::<i32, ()>::new();
        
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        let n4 = graph.add_node(4);
        
        graph.add_edge(n1, n2, ());
        graph.add_edge(n1, n3, ());
        graph.add_edge(n2, n4, ());
        graph.add_edge(n3, n4, ());
        
        // Check successors
        let succ_n1: Vec<_> = graph.successors(n1).collect();
        assert_eq!(succ_n1.len(), 2);
        assert!(succ_n1.contains(&n2));
        assert!(succ_n1.contains(&n3));
        
        let succ_n4: Vec<_> = graph.successors(n4).collect();
        assert_eq!(succ_n4.len(), 0); // n4 has no successors
        
        // Check predecessors
        let pred_n4: Vec<_> = graph.predecessors(n4).collect();
        assert_eq!(pred_n4.len(), 2);
        assert!(pred_n4.contains(&n2));
        assert!(pred_n4.contains(&n3));
        
        let pred_n1: Vec<_> = graph.predecessors(n1).collect();
        assert_eq!(pred_n1.len(), 0); // n1 has no predecessors
    }
    
    #[test]
    fn test_reverse() {
        let mut graph = DiGraph::<char, i32>::new();
        
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        
        graph.add_edge(a, b, 1);
        graph.add_edge(b, c, 2);
        
        // Create reversed graph
        let reversed = graph.reverse();
        
        // Check node count is the same
        assert_eq!(reversed.node_count(), graph.node_count());
        
        // Check edges are reversed
        assert!(reversed.has_edge(b, a));
        assert!(reversed.has_edge(c, b));
        assert!(!reversed.has_edge(a, b)); // Original direction
        assert!(!reversed.has_edge(b, c)); // Original direction
        
        // Check edge weights are preserved
        assert_eq!(reversed.get_edge_weight(b, a), Some(&1));
        assert_eq!(reversed.get_edge_weight(c, b), Some(&2));
    }
    
    #[test]
    fn test_adding_node_data_directly() {
        let mut graph = DiGraph::<String, i32>::new();
        
        // Add nodes directly with data
        let n1 = graph.add_node("Direct Node 1".to_string());
        let _n2 = graph.add_node("Direct Node 2".to_string());
        
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.get_node_data(n1), Some(&"Direct Node 1".to_string()));
    }
    
    // For the advanced functionality tests that depend on Graph methods,
    // we can write separate implementations that use DiGraph's API
    
    #[test]
    fn test_digraph_graph_operations() {
        let mut graph = DiGraph::<char, i32>::new();
        
        /*
        Create a graph like:
            A ---> B ---> C
            |       \
            v       v
            D ----> E
        */
        
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        let d = graph.add_node('D');
        let e = graph.add_node('E');
        
        graph.add_edge(a, b, 1);
        graph.add_edge(b, c, 1);
        graph.add_edge(a, d, 1);
        graph.add_edge(b, e, 1);
        graph.add_edge(d, e, 1);
        
        // Check basic operations
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 5);
        
        // Check node access
        assert_eq!(graph.get_node_data(a), Some(&'A'));
        assert_eq!(graph.get_node_data(e), Some(&'E'));
        
        // Check edge access
        assert_eq!(graph.get_edge_weight(a, b), Some(&1));
        assert_eq!(graph.get_edge_weight(d, e), Some(&1));
        assert_eq!(graph.get_edge_weight(c, d), None); // Non-existent edge
    }
    
    #[test]
    fn test_digraph_complex_paths() {
        let mut graph = DiGraph::<char, i32>::new();
        
        /*
        Create a graph like:
            A --2--> B --1--> C
            |        \       /
            |         3     2
            |          \   /
            5           v v
            |           D
            v          /
            E <---4----
        */
        
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        let d = graph.add_node('D');
        let e = graph.add_node('E');
        
        graph.add_edge(a, b, 2);
        graph.add_edge(b, c, 1);
        graph.add_edge(b, d, 3);
        graph.add_edge(c, d, 2);
        graph.add_edge(d, e, 4);
        graph.add_edge(a, e, 5);
        
        // Check path existence
        assert!(graph.has_edge(a, b));
        assert!(graph.has_edge(b, d));
        assert!(graph.has_edge(a, e));
        
        // Check path non-existence
        assert!(!graph.has_edge(e, a));
        assert!(!graph.has_edge(c, a));
    }
    
    #[test]
    fn test_subgraph() {
        let mut graph = DiGraph::<i32, &str>::new();
        
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        let n4 = graph.add_node(4);
        
        graph.add_edge(n1, n2, "1->2");
        graph.add_edge(n2, n3, "2->3");
        graph.add_edge(n1, n4, "1->4");
        graph.add_edge(n3, n4, "3->4");
        
        // Create subgraph with nodes 1, 2, and 4
        let subgraph = graph.subgraph(vec![n1, n2, n4]);
        
        // Check node count
        assert_eq!(subgraph.node_count(), 3);
        assert!(subgraph.get_node_data(n1).is_some());
        assert!(subgraph.get_node_data(n2).is_some());
        assert!(subgraph.get_node_data(n4).is_some());
        assert!(subgraph.get_node_data(n3).is_none());
        
        // Check edge count and edge existence
        assert_eq!(subgraph.edge_count(), 2); // Should have edges 1->2 and 1->4
        assert!(subgraph.has_edge(n1, n2));
        assert!(subgraph.has_edge(n1, n4));
        assert!(!subgraph.has_edge(n2, n3)); // Not in subgraph
        assert!(!subgraph.has_edge(n3, n4)); // Not in subgraph
        
        // Check edge weights
        assert_eq!(subgraph.get_edge_weight(n1, n2), Some(&"1->2"));
        assert_eq!(subgraph.get_edge_weight(n1, n4), Some(&"1->4"));
    }
    
    #[test]
    fn test_digraph_add_nodes_from() {
        let mut graph = DiGraph::<i32, ()>::new();
        let nodes = vec![1, 2, 3, 4];
        let keys = graph.add_nodes_from(nodes);
        
        assert_eq!(keys.len(), 4);
        assert_eq!(graph.node_count(), 4);
        
        for (i, &key) in keys.iter().enumerate() {
            assert_eq!(*graph.get_node_data(key).unwrap(), i as i32 + 1);
        }
    }
    
    #[test]
    fn test_digraph_add_edges_from() {
        let mut graph = DiGraph::<char, i32>::new();
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        
        let edges = vec![(a, b, 1), (b, c, 2), (c, a, 3)];
        let added = graph.add_edges_from(edges);
        
        assert_eq!(added, 3); // All edges should be added
        assert_eq!(graph.edge_count(), 3);
        
        assert!(graph.has_edge(a, b));
        assert!(graph.has_edge(b, c));
        assert!(graph.has_edge(c, a));
        
        assert_eq!(graph.get_edge_weight(a, b), Some(&1));
        assert_eq!(graph.get_edge_weight(b, c), Some(&2));
        assert_eq!(graph.get_edge_weight(c, a), Some(&3));
    }
    
    #[test]
    fn test_digraph_remove_edges() {
        let mut graph = DiGraph::<i32, &str>::new();
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        
        graph.add_edge(n1, n2, "a");
        graph.add_edge(n2, n3, "b");
        graph.add_edge(n3, n1, "c");
        
        let removed_edge = graph.remove_edge(n1, n2);
        assert_eq!(removed_edge, Some("a"));
        
        // Check edge counts and presence
        assert_eq!(graph.edge_count(), 2);
        assert!(!graph.has_edge(n1, n2));
        assert!(graph.has_edge(n2, n3));
        assert!(graph.has_edge(n3, n1));
        
        // Try removing a non-existent edge
        let non_existent = graph.remove_edge(n1, n2);
        assert_eq!(non_existent, None);
    }
    
    #[test]
    fn test_digraph_clear() {
        let mut graph = DiGraph::<i32, &str>::new();
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        
        graph.add_edge(n1, n2, "a");
        
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        
        graph.clear();
        
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }
    
    #[test]
    fn test_digraph_clear_edges() {
        let mut graph = DiGraph::<i32, &str>::new();
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        
        graph.add_edge(n1, n2, "a");
        
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        
        graph.clear_edges();
        
        assert_eq!(graph.node_count(), 2); // Nodes remain
        assert_eq!(graph.edge_count(), 0); // But edges are gone
    }
    
    #[test]
    fn test_digraph_complex_types() {
        #[derive(Clone, Debug)]
        struct User {
            name: String,
            age: u32,
        }
        
        #[derive(Clone, Debug)]
        struct Connection {
            strength: f64,
            timestamp: u64,
        }
        
        let mut graph = DiGraph::<User, Connection>::new();
        
        let user1 = graph.add_node(User { name: "Alice".to_string(), age: 25 });
        let user2 = graph.add_node(User { name: "Bob".to_string(), age: 30 });
        
        let connection = Connection { strength: 0.8, timestamp: 1234567890 };
        graph.add_edge(user1, user2, connection);
        
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert!(graph.has_edge(user1, user2));
        assert!(!graph.has_edge(user2, user1));
        
        // Check node data
        if let Some(user) = graph.get_node_data(user1) {
            assert_eq!(user.name, "Alice");
            assert_eq!(user.age, 25);
        }
        
        // Check edge data
        if let Some(conn) = graph.get_edge_weight(user1, user2) {
            assert_eq!(conn.strength, 0.8);
            assert_eq!(conn.timestamp, 1234567890);
        }
    }
    
    #[test]
    fn test_digraph_display() {
        let mut graph = DiGraph::<i32, i32>::with_name("Test DiGraph");
        graph.add_node(1);
        graph.add_node(2);
        
        let display_string = format!("{}", graph);
        assert!(display_string.contains("Test DiGraph"));
        assert!(display_string.contains("nodes=2"));
        assert!(display_string.contains("edges=0"));
    }
} 