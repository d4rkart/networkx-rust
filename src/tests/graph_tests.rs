use crate::Graph;
use crate::NodeKey;
use std::fmt::Debug;

#[cfg(test)]
mod graph_tests {
    use super::*;
    
    // Basic functionality tests
    
    #[test]
    fn test_new_graph() {
        let graph = Graph::<i32, f64>::new(true);
        assert!(graph.is_directed());
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        
        let graph = Graph::<String, i32>::new(false);
        assert!(!graph.is_directed());
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }
    
    #[test]
    fn test_basic_node_operations() {
        let mut graph = Graph::<String, i32>::new(true);
        
        // Add nodes
        let n1 = graph.add_node("Node 1".to_string());
        let n2 = graph.add_node("Node 2".to_string());
        
        // Check node count
        assert_eq!(graph.node_count(), 2);
        
        // Check node existence
        assert!(graph.has_node(n1));
        assert!(graph.has_node(n2));
        assert!(!graph.has_node(999)); // Non-existent node
        
        // Check node data
        assert_eq!(graph.get_node_data(n1), Some(&"Node 1".to_string()));
        assert_eq!(graph.get_node_data(n2), Some(&"Node 2".to_string()));
        assert_eq!(graph.get_node_data(999), None); // Non-existent node
        
        // Check node mutation
        if let Some(data) = graph.get_node_data_mut(n1) {
            *data = "Updated Node 1".to_string();
        }
        assert_eq!(graph.get_node_data(n1), Some(&"Updated Node 1".to_string()));
        
        // Remove node
        assert_eq!(graph.remove_node(n1), Some("Updated Node 1".to_string()));
        assert_eq!(graph.node_count(), 1);
        assert!(!graph.has_node(n1));
        assert!(graph.has_node(n2));
        
        // Try to remove non-existent node
        assert_eq!(graph.remove_node(n1), None);
    }
    
    #[test]
    fn test_directed_edge_operations() {
        let mut graph = Graph::<i32, String>::new(true);
        
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
        assert_eq!(graph.get_edge_weight(n1, n3), None); // Non-existent edge
        
        // Update edge
        assert!(!graph.add_edge(n1, n2, "Updated Edge 1->2".to_string())); // Returns false for update
        assert_eq!(graph.get_edge_weight(n1, n2), Some(&"Updated Edge 1->2".to_string()));
        
        // Remove edge
        assert_eq!(graph.remove_edge(n1, n2), Some("Updated Edge 1->2".to_string()));
        assert_eq!(graph.edge_count(), 1);
        assert!(!graph.has_edge(n1, n2));
        assert!(graph.has_edge(n2, n3));
        
        // Try to remove non-existent edge
        assert_eq!(graph.remove_edge(n1, n2), None);
    }
    
    #[test]
    fn test_undirected_edge_operations() {
        let mut graph = Graph::<i32, String>::new(false);
        
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        
        // Add edge
        assert!(graph.add_edge(n1, n2, "Edge 1-2".to_string()));
        
        // Check edge count - should be 1 even though accessible both ways
        assert_eq!(graph.edge_count(), 1);
        
        // Check edge existence both ways
        assert!(graph.has_edge(n1, n2));
        assert!(graph.has_edge(n2, n1)); // Both directions should exist
        
        // Check edge data both ways
        assert_eq!(graph.get_edge_weight(n1, n2), Some(&"Edge 1-2".to_string()));
        assert_eq!(graph.get_edge_weight(n2, n1), Some(&"Edge 1-2".to_string()));
        
        // Update edge via either direction
        assert!(!graph.add_edge(n2, n1, "Updated Edge 1-2".to_string())); // Update from opposite direction
        assert_eq!(graph.get_edge_weight(n1, n2), Some(&"Updated Edge 1-2".to_string()));
        assert_eq!(graph.get_edge_weight(n2, n1), Some(&"Updated Edge 1-2".to_string()));
        
        // Remove edge via either direction
        assert_eq!(graph.remove_edge(n2, n1), Some("Updated Edge 1-2".to_string()));
        assert_eq!(graph.edge_count(), 0);
        assert!(!graph.has_edge(n1, n2));
        assert!(!graph.has_edge(n2, n1));
    }
    
    #[test]
    fn test_node_removal_cascade() {
        let mut graph = Graph::<i32, i32>::new(true);
        
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        
        graph.add_edge(n1, n2, 12);
        graph.add_edge(n2, n3, 23);
        graph.add_edge(n3, n1, 31);
        
        assert_eq!(graph.edge_count(), 3);
        
        // Remove node and check that its edges are also removed
        graph.remove_node(n2);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1); // Only edge n3->n1 should remain
        assert!(!graph.has_edge(n1, n2));
        assert!(!graph.has_edge(n2, n3));
        assert!(graph.has_edge(n3, n1));
    }
    
    // Advanced functionality tests
    
    #[test]
    fn test_dfs_directed() {
        let mut graph = Graph::<char, i32>::new(true);
        
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
        
        // DFS from A should visit all nodes
        let dfs_result = graph.dfs(a);
        assert_eq!(dfs_result.len(), 5);
        assert_eq!(dfs_result[0], a); // DFS always starts with the start node
        
        // DFS from B should not include A or D
        let dfs_result = graph.dfs(b);
        assert_eq!(dfs_result.len(), 3);
        assert_eq!(dfs_result[0], b);
        assert!(dfs_result.contains(&c));
        assert!(dfs_result.contains(&e));
        assert!(!dfs_result.contains(&a));
        assert!(!dfs_result.contains(&d));
    }
    
    #[test]
    fn test_bfs_directed() {
        let mut graph = Graph::<char, i32>::new(true);
        
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
        
        // BFS from A should have a specific order
        let bfs_result = graph.bfs(a);
        assert_eq!(bfs_result.len(), 5);
        assert_eq!(bfs_result[0], a); // BFS always starts with the start node
        
        // Depth 1 nodes (b and d) should come before depth 2 nodes (c and e)
        let b_pos = bfs_result.iter().position(|&x| x == b).unwrap();
        let d_pos = bfs_result.iter().position(|&x| x == d).unwrap();
        let c_pos = bfs_result.iter().position(|&x| x == c).unwrap();
        let e_pos = bfs_result.iter().position(|&x| x == e).unwrap();
        
        assert!(b_pos < c_pos); // B comes before its child C
        assert!(b_pos < e_pos); // B comes before its child E
        assert!(d_pos < e_pos); // D comes before its child E
    }
    
    #[test]
    fn test_shortest_path_with_cycles() {
        let mut graph = Graph::<char, i32>::new(true);
        
        /*
        Create a graph like:
            A --2--> B --1--> C
            ^        \       /
            |         3     2
            |          \   /
            |           v v
            '-----5----- D
        */
        
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        let d = graph.add_node('D');
        
        graph.add_edge(a, b, 2);
        graph.add_edge(b, c, 1);
        graph.add_edge(b, d, 3);
        graph.add_edge(c, d, 2);
        graph.add_edge(d, a, 5);
        
        // Test shortest path from A to D
        let (path, cost) = graph.shortest_path(a, d, |&w| w as f64).unwrap();
        assert_eq!(path.len(), 3); // A -> B -> D
        assert_eq!(cost, 5.0); // 2 + 3 = 5
        
        // Test shortest path from D to B
        let (path, cost) = graph.shortest_path(d, b, |&w| w as f64).unwrap();
        assert_eq!(path.len(), 3); // D -> A -> B
        assert_eq!(cost, 7.0); // 5 + 2 = 7
        
        // Verify path contents - manually for the test
        let node_data: Vec<char> = path.iter().map(|&k| *graph.get_node_data(k).unwrap()).collect();
        assert_eq!(node_data, vec!['D', 'A', 'B']);
    }
    
    #[test]
    fn test_nbunch_iter_edge_cases() {
        let mut graph = Graph::<i32, ()>::new(true);
        let n1 = graph.add_node(1);
        let _n2 = graph.add_node(2);
        
        // Test with empty nbunch
        let empty_vec: Vec<NodeKey> = Vec::new();
        let nodes = graph.nbunch_iter(Some(empty_vec));
        assert_eq!(nodes.len(), 0);
        
        // Test with non-existent nodes
        let nodes = graph.nbunch_iter(Some(vec![999, 1000]));
        assert_eq!(nodes.len(), 0);
        
        // Test with mix of existing and non-existing nodes
        let nodes = graph.nbunch_iter(Some(vec![n1, 999]));
        assert_eq!(nodes.len(), 1);
        assert!(nodes.contains(&n1));
    }
    
    #[test]
    fn test_find_nodes_complex() {
        #[derive(Clone, Debug)]
        struct Person {
            name: String,
            age: u32,
        }
        
        let mut graph = Graph::<Person, ()>::new(true);
        let p1 = graph.add_node(Person { name: "Alice".to_string(), age: 25 });
        let _p2 = graph.add_node(Person { name: "Bob".to_string(), age: 30 });
        let p3 = graph.add_node(Person { name: "Charlie".to_string(), age: 35 });
        let p4 = graph.add_node(Person { name: "Diana".to_string(), age: 25 });
        
        // Find by name
        let a_nodes = graph.find_nodes(|p| p.name.starts_with("A"));
        assert_eq!(a_nodes.len(), 1);
        assert!(a_nodes.contains(&p1));
        
        // Find by age
        let age_25_nodes = graph.find_nodes(|p| p.age == 25);
        assert_eq!(age_25_nodes.len(), 2);
        assert!(age_25_nodes.contains(&p1));
        assert!(age_25_nodes.contains(&p4));
        
        // Find by complex criteria
        let complex_nodes = graph.find_nodes(|p| p.name.len() > 5 && p.age > 30);
        assert_eq!(complex_nodes.len(), 1);
        assert!(complex_nodes.contains(&p3));
    }
    
    #[test]
    fn test_add_edges_from_edge_cases() {
        let mut graph = Graph::<i32, i32>::new(true);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        
        // Test with empty edges list
        let empty_edges: Vec<(NodeKey, NodeKey, i32)> = Vec::new();
        assert_eq!(graph.add_edges_from(empty_edges), 0);
        
        // Test with duplicate edges
        let edges = vec![(n1, n2, 1), (n1, n2, 2)];
        assert_eq!(graph.add_edges_from(edges), 1); // Only counts as one new edge
        assert_eq!(graph.get_edge_weight(n1, n2), Some(&2)); // Last value overwrites
        
        // Test with non-existent nodes
        let edges = vec![(n1, 999, 3)]; // 999 doesn't exist
        assert_eq!(graph.add_edges_from(edges), 0); // Should fail
        
        // Check final edge count
        assert_eq!(graph.edge_count(), 1);
    }
    
    // #[test]
    // fn test_to_directed_and_undirected() {
    //     // Test undirected to directed conversion
    //     let mut undirected = Graph::<char, i32>::new(false);
    //     let a = undirected.add_node('A');
    //     let b = undirected.add_node('B');
    //     let c = undirected.add_node('C');
        
    //     // Add edges in the undirected graph
    //     undirected.add_edge(a, b, 1);
    //     undirected.add_edge(b, c, 2);
        
    //     // Verify initial state of undirected graph
    //     assert!(!undirected.is_directed());
    //     assert_eq!(undirected.node_count(), 3);
        
    //     // Undirected graph has 2 edges, but they're accessible in both directions
    //     // Check edges by enumerating them
    //     let edges = undirected.edges();
    //     assert_eq!(edges.len(), 2);
        
    //     // In undirected graph, edges exist in both directions
    //     assert!(undirected.has_edge(a, b) && undirected.has_edge(b, a));
    //     assert!(undirected.has_edge(b, c) && undirected.has_edge(c, b));
        
    //     // Convert to directed
    //     let directed = undirected.to_directed();
        
    //     // Verify directed graph properties
    //     assert!(directed.is_directed());
    //     assert_eq!(directed.node_count(), 3);
        
    //     // Inspect all edges in the directed graph
    //     eprintln!("Directed graph edges: {:?}", directed.edges());
        
    //     // Check individual edges now exist in both directions
    //     assert!(directed.has_edge(a, b));
    //     assert!(directed.has_edge(b, a));
    //     assert!(directed.has_edge(b, c));
    //     assert!(directed.has_edge(c, b));
        
    //     // In a directed graph converted from undirected, we now have separate edges
    //     // for each direction, so the total should be double (except for self-loops)
    //     let edge_count = directed.edge_count();
    //     eprintln!("Directed edge count: {}", edge_count);
    //     assert_eq!(edge_count, 4);
        
    //     // Test directed to undirected conversion
    //     let mut directed_original = Graph::<char, i32>::new(true);
    //     let a = directed_original.add_node('A');
    //     let b = directed_original.add_node('B');
    //     let c = directed_original.add_node('C');
        
    //     // Add one-way edges in the directed graph
    //     directed_original.add_edge(a, b, 1);
    //     directed_original.add_edge(b, c, 2);
    //     directed_original.add_edge(c, a, 3);
        
    //     // Verify initial state of directed graph
    //     assert!(directed_original.is_directed());
    //     assert_eq!(directed_original.node_count(), 3);
    //     assert_eq!(directed_original.edge_count(), 3);
        
    //     // In directed graph, edges exist only in the specified direction
    //     assert!(directed_original.has_edge(a, b) && !directed_original.has_edge(b, a));
    //     assert!(directed_original.has_edge(b, c) && !directed_original.has_edge(c, b));
    //     assert!(directed_original.has_edge(c, a) && !directed_original.has_edge(a, c));
        
    //     // Convert to undirected
    //     let undirected_result = directed_original.to_undirected();
        
    //     // Verify undirected graph properties
    //     assert!(!undirected_result.is_directed());
    //     assert_eq!(undirected_result.node_count(), 3);
        
    //     // Count edges to verify
    //     let edge_list = undirected_result.edges();
    //     assert_eq!(edge_list.len(), 3);
        
    //     // All edges should be bidirectional now
    //     assert!(undirected_result.has_edge(a, b) && undirected_result.has_edge(b, a));
    //     assert!(undirected_result.has_edge(b, c) && undirected_result.has_edge(c, b));
    //     assert!(undirected_result.has_edge(c, a) && undirected_result.has_edge(a, c));
        
    //     // Test self-loops handling
    //     let mut graph_with_loops = Graph::<i32, i32>::new(false);
    //     let n1 = graph_with_loops.add_node(1);
    //     let n2 = graph_with_loops.add_node(2);
        
    //     // Add a normal edge and a self-loop
    //     graph_with_loops.add_edge(n1, n2, 10);
    //     graph_with_loops.add_edge(n1, n1, 5); // Self-loop
        
    //     let loop_edges = graph_with_loops.edges();
    //     assert_eq!(loop_edges.len(), 2);
        
    //     // Convert to directed
    //     let directed_loops = graph_with_loops.to_directed();
        
    //     // The normal edge should be duplicated, but the self-loop should not
    //     assert!(directed_loops.has_edge(n1, n2) && directed_loops.has_edge(n2, n1));
    //     assert!(directed_loops.has_edge(n1, n1));
    //     assert_eq!(directed_loops.edge_count(), 3); // 2 for n1<->n2 + 1 for self-loop
    // }
    
    #[test]
    fn test_graph_string_operators() {
        let mut graph = Graph::<i32, i32>::with_name(true, "Test Graph");
        graph.add_node(1);
        graph.add_node(2);
        
        // Test display
        let display_string = format!("{}", graph);
        assert!(display_string.contains("Test Graph"));
        assert!(display_string.contains("nodes=2"));
        
        // Test debug
        let debug_string = format!("{:?}", graph);
        assert!(debug_string.contains("Graph"));
    }
    
    #[test]
    fn test_node_iteration() {
        let mut graph = Graph::<i32, i32>::new(true);
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        
        // Test node existence
        let all_nodes = graph.nodes();
        assert_eq!(all_nodes.len(), 3);
        assert!(all_nodes.contains(&n1));
        assert!(all_nodes.contains(&n2));
        assert!(all_nodes.contains(&n3));
        
        // Test neighbor iteration
        graph.add_edge(n1, n2, 12);
        graph.add_edge(n1, n3, 13);
        
        let neighbors: Vec<_> = graph.neighbors(n1).into_iter().collect();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&n2));
        assert!(neighbors.contains(&n3));
        
        // Test edges iteration
        let edges: Vec<_> = graph.edges().into_iter().collect();
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(n1, n2, 12)));
        assert!(edges.contains(&(n1, n3, 13)));
    }
} 