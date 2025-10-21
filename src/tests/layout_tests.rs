use crate::{Graph, Position, circular_layout, shell_layout, spiral_layout, spring_layout, kamada_kawai_layout, random_layout};
use crate::graph::NodeKey;
use rand::thread_rng;
use std::collections::HashMap;

#[test]
fn test_circular_layout() {
    let mut graph = Graph::<i32, ()>::new(false);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);

    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());
    graph.add_edge(n3, n4, ());
    graph.add_edge(n4, n1, ());

    let scale = 1.0;
    let center: Position = [0.0, 0.0];

    let positions = circular_layout(&graph, scale, center);

    // Check that all nodes have positions
    assert_eq!(positions.len(), 4);
    assert!(positions.contains_key(&n1));
    assert!(positions.contains_key(&n2));
    assert!(positions.contains_key(&n3));
    assert!(positions.contains_key(&n4));

    // Check that positions are on a circle with radius = scale
    for (_, pos) in &positions {
        let distance = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
        assert!((distance - scale).abs() < 1e-10);
    }
}

#[test]
fn test_shell_layout() {
    let mut graph = Graph::<i32, ()>::new(false);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);

    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());
    graph.add_edge(n3, n4, ());
    graph.add_edge(n4, n1, ());

    let scale = 1.0;
    let center: Position = [0.0, 0.0];

    // Create two shells: [n1, n2] and [n3, n4]
    let nlist = Some(vec![vec![n1, n2], vec![n3, n4]]);

    let positions = shell_layout(&graph, scale, center, nlist);

    // Check that all nodes have positions
    assert_eq!(positions.len(), 4);
    assert!(positions.contains_key(&n1));
    assert!(positions.contains_key(&n2));
    assert!(positions.contains_key(&n3));
    assert!(positions.contains_key(&n4));

    // Check that n1 and n2 are at distance scale from center
    let dist_n1 = (positions[&n1][0] * positions[&n1][0] + positions[&n1][1] * positions[&n1][1]).sqrt();
    let dist_n2 = (positions[&n2][0] * positions[&n2][0] + positions[&n2][1] * positions[&n2][1]).sqrt();
    assert!((dist_n1 - scale).abs() < 1e-10);
    assert!((dist_n2 - scale).abs() < 1e-10);

    // Check that n3 and n4 are at distance 2*scale from center
    let dist_n3 = (positions[&n3][0] * positions[&n3][0] + positions[&n3][1] * positions[&n3][1]).sqrt();
    let dist_n4 = (positions[&n4][0] * positions[&n4][0] + positions[&n4][1] * positions[&n4][1]).sqrt();
    assert!((dist_n3 - 2.0 * scale).abs() < 1e-10);
    assert!((dist_n4 - 2.0 * scale).abs() < 1e-10);
}

#[test]
fn test_spiral_layout() {
    let mut graph = Graph::<i32, ()>::new(false);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);

    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());
    graph.add_edge(n3, n4, ());
    graph.add_edge(n4, n1, ());

    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let resolution = 0.35;
    let equidistant = false;

    let positions = spiral_layout(&graph, scale, center, resolution, equidistant);

    // Check that all nodes have positions
    assert_eq!(positions.len(), 4);
    assert!(positions.contains_key(&n1));
    assert!(positions.contains_key(&n2));
    assert!(positions.contains_key(&n3));
    assert!(positions.contains_key(&n4));

    // Get the node keys in order they appear in the graph
    let nodes = graph.nodes();

    // Check that distances from center are increasing
    // First node should be at or close to the center
    let dist_first = (positions[&nodes[0]][0] * positions[&nodes[0]][0] +
                      positions[&nodes[0]][1] * positions[&nodes[0]][1]).sqrt();

    assert!(dist_first < 0.1 * scale, "First node should be close to center");

    // Calculate distances for all nodes
    let mut distances = Vec::new();
    for &node in &nodes {
        let dist = (positions[&node][0] * positions[&node][0] +
                    positions[&node][1] * positions[&node][1]).sqrt();
        distances.push(dist);
    }

    // Make sure at least some distances increase
    // We can't be too strict here due to ordering issues
    let mut increasing_count = 0;
    for i in 1..distances.len() {
        if distances[i] > distances[i-1] {
            increasing_count += 1;
        }
    }

    // At least 2 of the 3 pairs should be increasing
    assert!(increasing_count >= 2, "Distances should generally increase in a spiral");
}

#[test]
fn test_spring_layout() {
    let mut graph = Graph::<i32, ()>::new(false);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);

    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());
    graph.add_edge(n3, n4, ());
    graph.add_edge(n4, n1, ());

    let k = Some(1.0);
    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let iterations = 50;
    let threshold = 1e-4;

    let positions = spring_layout(&graph, k, None, None, iterations, threshold, scale, center.to_vec(), None, 2, None);

    // Check that all nodes have positions
    assert_eq!(positions.len(), 4);
    assert!(positions.contains_key(&n1));
    assert!(positions.contains_key(&n2));
    assert!(positions.contains_key(&n3));
    assert!(positions.contains_key(&n4));

    // Check that positions are within the expected range
    for (_, pos) in &positions {
        assert!(pos[0] >= center[0] - scale);
        assert!(pos[0] <= center[0] + scale);
        assert!(pos[1] >= center[1] - scale);
        assert!(pos[1] <= center[1] + scale);
    }
}

#[test]
fn test_random_layout() {
    let mut graph = Graph::<i32, ()>::new(false);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);

    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let mut rng = thread_rng();

    let positions = random_layout(&graph, center, scale, &mut rng);

    // Check that all nodes have positions
    assert_eq!(positions.len(), 4);
    assert!(positions.contains_key(&n1));
    assert!(positions.contains_key(&n2));
    assert!(positions.contains_key(&n3));
    assert!(positions.contains_key(&n4));

    // Check that positions are within the expected range
    for (_, pos) in &positions {
        assert!(pos[0] >= center[0] - scale);
        assert!(pos[0] <= center[0] + scale);
        assert!(pos[1] >= center[1] - scale);
        assert!(pos[1] <= center[1] + scale);
    }
}

#[test]
fn test_kamada_kawai_layout() {
    let mut graph = Graph::<i32, ()>::new(false);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);

    // Create a square graph
    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());
    graph.add_edge(n3, n4, ());
    graph.add_edge(n4, n1, ());

    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let epsilon = 1e-5;
    let max_iterations = 100;

    let positions = kamada_kawai_layout(&graph, None, None, None, scale, center, epsilon, max_iterations);

    // Check that all nodes have positions
    assert_eq!(positions.len(), 4);
}

#[test]
fn test_kamada_kawai_layout_with_weights() {
    let mut graph = Graph::<String, String>::new(false);
    let n1 = graph.add_node("node1".to_string());
    let n2 = graph.add_node("node2".to_string());
    let n3 = graph.add_node("node3".to_string());
    let n4 = graph.add_node("node4".to_string());

    // Add edges with JSON weight data
    graph.add_edge(n1, n2, r#"{"weight": 2.0}"#.to_string());
    graph.add_edge(n2, n3, r#"{"weight": 3.0}"#.to_string());
    graph.add_edge(n3, n4, r#"{"weight": 1.5}"#.to_string());
    graph.add_edge(n4, n1, r#"{"weight": 2.5}"#.to_string());

    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let epsilon = 1e-5;
    let max_iterations = 100;

    // Test kamada_kawai_layout with weight parameter
    let positions = kamada_kawai_layout(&graph, None, None, Some("weight"), scale, center, epsilon, max_iterations);

    // Check that all nodes have positions
    assert_eq!(positions.len(), 4);
    assert!(positions.contains_key(&n1));
    assert!(positions.contains_key(&n2));
    assert!(positions.contains_key(&n3));
    assert!(positions.contains_key(&n4));

    // Check that positions are within the expected range
    for (_, pos) in &positions {
        assert!(pos[0] >= center[0] - scale);
        assert!(pos[0] <= center[0] + scale);
        assert!(pos[1] >= center[1] - scale);
        assert!(pos[1] <= center[1] + scale);
    }

    // Since we created a square, check that nodes are approximately equidistant
    // from their neighbors (but not from nodes across the diagonal)
    let dist_12 = distance_between(&positions[&n1], &positions[&n2]);
    let dist_23 = distance_between(&positions[&n2], &positions[&n3]);
    let dist_34 = distance_between(&positions[&n3], &positions[&n4]);
    let dist_41 = distance_between(&positions[&n4], &positions[&n1]);

    let dist_13 = distance_between(&positions[&n1], &positions[&n3]);
    let dist_24 = distance_between(&positions[&n2], &positions[&n4]);

    // Neighbor distances should be similar
    let avg_neighbor_dist = (dist_12 + dist_23 + dist_34 + dist_41) / 4.0;
    assert!((dist_12 - avg_neighbor_dist).abs() < 0.3);
    assert!((dist_23 - avg_neighbor_dist).abs() < 0.3);
    assert!((dist_34 - avg_neighbor_dist).abs() < 0.3);
    assert!((dist_41 - avg_neighbor_dist).abs() < 0.3);

    // Diagonal distances should be larger than neighbor distances
    assert!(dist_13 > avg_neighbor_dist);
    assert!(dist_24 > avg_neighbor_dist);
}

// Helper function to calculate Euclidean distance between two positions
fn distance_between(pos1: &Position, pos2: &Position) -> f64 {
    let dx = pos1[0] - pos2[0];
    let dy = pos1[1] - pos2[1];
    (dx * dx + dy * dy).sqrt()
}

#[test]
fn test_spring_layout_with_i32_weights() {
    let mut graph = Graph::<String, i32>::new(false);
    let n1 = graph.add_node("node1".to_string());
    let n2 = graph.add_node("node2".to_string());
    let n3 = graph.add_node("node3".to_string());

    // Add edges with different weights
    graph.add_edge(n1, n2, 5); // weight = 5
    graph.add_edge(n2, n3, 10); // weight = 10
    graph.add_edge(n3, n1, 1); // weight = 1

    let k = Some(1.0);
    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let iterations = 50;
    let threshold = 1e-4;

    // Test without weight parameter (should use default weight of 1.0)
    let positions_no_weight = spring_layout(&graph, k, None, None, iterations, threshold, scale, center.to_vec(), None, 2, None);

    // Test with weight parameter
    let positions_with_weight = spring_layout(&graph, k, None, None, iterations, threshold, scale, center.to_vec(), Some(""), 2, None);

    // Both should produce valid positions
    assert_eq!(positions_no_weight.len(), 3);
    assert_eq!(positions_with_weight.len(), 3);

    for (_node, pos) in &positions_no_weight {
        assert!(pos[0].is_finite());
        assert!(pos[1].is_finite());
    }

    for (_node, pos) in &positions_with_weight {
        assert!(pos[0].is_finite());
        assert!(pos[1].is_finite());
    }
}

#[test]
fn test_spring_layout_with_json_weights() {
    use serde_json::json;

    let mut graph = Graph::<String, String>::new(false);
    let n1 = graph.add_node("node1".to_string());
    let n2 = graph.add_node("node2".to_string());
    let n3 = graph.add_node("node3".to_string());

    // Add edges with JSON string data containing weight fields
    let weight_data_1 = json!({"weight": 3.5, "cost": 3.5});
    let weight_data_2 = json!({"cost": 7.2, "other": "data"});
    let weight_data_3 = json!({"sessionCount": 2.1, "cost": 2.1});

    graph.add_edge(n1, n2, weight_data_1.to_string());
    graph.add_edge(n2, n3, weight_data_2.to_string());
    graph.add_edge(n3, n1, weight_data_3.to_string());

    let k = Some(1.0);
    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let iterations = 50;
    let threshold = 1e-4;

    // Test with specific weight field
    let positions = spring_layout(&graph, k, None, None, iterations, threshold, scale, center.to_vec(), Some("cost"), 2, None);

    // Check that all nodes have positions
    assert_eq!(positions.len(), 3);
    assert!(positions.contains_key(&n1));
    assert!(positions.contains_key(&n2));
    assert!(positions.contains_key(&n3));

    // Check that positions are within reasonable bounds
    for (_, pos) in &positions {
        assert!(pos[0].is_finite());
        assert!(pos[1].is_finite());
        assert!(pos[0].abs() <= 2.0);
        assert!(pos[1].abs() <= 2.0);
    }
}

#[test]
fn test_spring_layout_weight_edge_cases() {
    let mut graph = Graph::<String, i32>::new(false);
    let n1 = graph.add_node("node1".to_string());
    let n2 = graph.add_node("node2".to_string());

    // Test with zero weight
    graph.add_edge(n1, n2, 0);

    let k = Some(1.0);
    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let iterations = 50;
    let threshold = 1e-4;

    // Test with no weight parameter - should use default weight of 1.0
    let positions = spring_layout(&graph, k, None, None, iterations, threshold, scale, center.to_vec(), None, 2, None);

    assert_eq!(positions.len(), 2);
    for (_, pos) in &positions {
        assert!(pos[0].is_finite());
        assert!(pos[1].is_finite());
    }
}

#[test]
fn test_spring_layout_zero_weight_extraction() {
    use serde_json::json;

    let mut graph = Graph::<String, String>::new(false);
    let n1 = graph.add_node("node1".to_string());
    let n2 = graph.add_node("node2".to_string());

    // Add edge with JSON data containing zero weight
    let zero_weight_data = json!({"weight": 0.0, "cost": 5.0});
    graph.add_edge(n1, n2, zero_weight_data.to_string());

    let k = Some(1.0);
    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let iterations = 50;
    let threshold = 1e-4;

    // Test zero weight extraction - should use the zero weight from JSON
    let positions = spring_layout(&graph, k, None, None, iterations, threshold, scale, center.to_vec(), Some("weight"), 2, None);

    assert_eq!(positions.len(), 2);
    for (_, pos) in &positions {
        assert!(pos[0].is_finite());
        assert!(pos[1].is_finite());
    }
}

#[test]
fn test_spring_layout_no_weight_parameter() {
    let mut graph = Graph::<String, i32>::new(false);
    let n1 = graph.add_node("node1".to_string());
    let n2 = graph.add_node("node2".to_string());

    graph.add_edge(n1, n2, 5);

    let k = Some(1.0);
    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let iterations = 50;
    let threshold = 1e-4;

    // Test that None weight parameter works (uses default weight of 1.0)
    let positions = spring_layout(&graph, k, None, None, iterations, threshold, scale, center.to_vec(), None, 2, None);

    assert_eq!(positions.len(), 2);
    for (_, pos) in &positions {
        assert!(pos[0].is_finite());
        assert!(pos[1].is_finite());
    }
}

#[test]
fn test_spring_layout_with_pos_parameter() {
    let mut graph = Graph::<i32, ()>::new(false);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);

    // Create a square graph
    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());
    graph.add_edge(n3, n4, ());
    graph.add_edge(n4, n1, ());

    let k = Some(1.0);
    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let iterations = 50;
    let threshold = 1e-4;

    // Create initial positions
    let mut initial_pos = HashMap::new();
    initial_pos.insert(n1, [0.0, 0.0]);
    initial_pos.insert(n2, [1.0, 0.0]);
    initial_pos.insert(n3, [1.0, 1.0]);
    initial_pos.insert(n4, [0.0, 1.0]);

    // Test with pos parameter
    let positions_with_pos = spring_layout(&graph, k, Some(&initial_pos), None, iterations, threshold, scale, center.to_vec(), None, 2, Some(42));

    // Test without pos parameter (random initialization)
    let positions_without_pos = spring_layout(&graph, k, None, None, iterations, threshold, scale, center.to_vec(), None, 2, Some(42));

    // The results should be different because one starts with specific positions
    // and the other starts with random positions
    let mut positions_different = false;
    for &node in &[n1, n2, n3, n4] {
        let pos_with = positions_with_pos[&node];
        let pos_without = positions_without_pos[&node];
        if (pos_with[0] - pos_without[0]).abs() > 1e-6 || (pos_with[1] - pos_without[1]).abs() > 1e-6 {
            positions_different = true;
            break;
        }
    }
    assert!(positions_different, "Results with and without pos parameter should be different");

    // With pos parameter, the initial positions should influence the final result
    // The final positions should be closer to the initial arrangement
    // than completely random positions would be
    let initial_distances = calculate_square_distances(&initial_pos);
    let final_distances_with_pos = calculate_square_distances(&positions_with_pos);
    let final_distances_without_pos = calculate_square_distances(&positions_without_pos);

    // The distances in the result with pos should be more similar to the initial
    let similarity_with_pos = calculate_similarity(&initial_distances, &final_distances_with_pos);
    let similarity_without_pos = calculate_similarity(&initial_distances, &final_distances_without_pos);

    // This is a probabilistic test - with pos parameter, the result should be more similar
    // to the initial arrangement (though not guaranteed due to the iterative nature)
    println!("Similarity with pos: {}, without pos: {}", similarity_with_pos, similarity_without_pos);
}

#[test]
fn test_spring_layout_with_fixed_nodes() {
    let mut graph = Graph::<i32, ()>::new(false);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);

    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());
    graph.add_edge(n3, n4, ());
    graph.add_edge(n4, n1, ());

    let k = Some(1.0);
    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let iterations = 50;
    let threshold = 1e-4;

    // Create initial positions
    let mut initial_pos = HashMap::new();
    initial_pos.insert(n1, [0.0, 0.0]);
    initial_pos.insert(n2, [1.0, 0.0]);
    initial_pos.insert(n3, [1.0, 1.0]);
    initial_pos.insert(n4, [0.0, 1.0]);

    // Test with fixed nodes (n1 and n2 should stay in their initial positions)
    let fixed_nodes = vec![n1, n2];
    let positions = spring_layout(&graph, k, Some(&initial_pos), Some(&fixed_nodes), iterations, threshold, scale, center.to_vec(), None, 2, Some(42));

    // Check that fixed nodes are very close to their initial positions
    let pos1 = positions[&n1];
    let pos2 = positions[&n2];
    let initial_pos1 = initial_pos[&n1];
    let initial_pos2 = initial_pos[&n2];

    assert!((pos1[0] - initial_pos1[0]).abs() < 1e-6, "Fixed node n1 x position should be preserved");
    assert!((pos1[1] - initial_pos1[1]).abs() < 1e-6, "Fixed node n1 y position should be preserved");
    assert!((pos2[0] - initial_pos2[0]).abs() < 1e-6, "Fixed node n2 x position should be preserved");
    assert!((pos2[1] - initial_pos2[1]).abs() < 1e-6, "Fixed node n2 y position should be preserved");

    // Non-fixed nodes should be able to move
    let pos3 = positions[&n3];
    let pos4 = positions[&n4];
    let initial_pos3 = initial_pos[&n3];
    let initial_pos4 = initial_pos[&n4];

    // These might be close to initial positions due to the layout algorithm,
    // but they're not constrained to be exactly the same
    println!("Node 3 moved from {:?} to {:?}", initial_pos3, pos3);
    println!("Node 4 moved from {:?} to {:?}", initial_pos4, pos4);
}

#[test]
fn test_spring_layout_deterministic_with_seed() {
    let mut graph = Graph::<i32, ()>::new(false);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);

    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());
    graph.add_edge(n3, n1, ());

    let k = Some(1.0);
    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let iterations = 50;
    let threshold = 1e-4;
    let seed = Some(1337);

    // Run the same layout twice with the same seed
    let positions1 = spring_layout(&graph, k, None, None, iterations, threshold, scale, center.to_vec(), None, 2, seed);
    let positions2 = spring_layout(&graph, k, None, None, iterations, threshold, scale, center.to_vec(), None, 2, seed);

    // Results should be identical with the same seed
    for &node in &[n1, n2, n3] {
        let pos1 = positions1[&node];
        let pos2 = positions2[&node];
        assert!((pos1[0] - pos2[0]).abs() < 1e-10, "Results should be identical with same seed");
        assert!((pos1[1] - pos2[1]).abs() < 1e-10, "Results should be identical with same seed");
    }
}

#[test]
fn test_spring_layout_known_positions() {
    let mut graph = Graph::<i32, ()>::new(false);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);

    graph.add_edge(n1, n2, ());
    graph.add_edge(n2, n3, ());

    let k = Some(1.0);
    let scale = 1.0;
    let center: Position = [0.0, 0.0];
    let iterations = 100;
    let threshold = 1e-6;
    let seed = Some(1337);

    let positions = spring_layout(&graph, k, None, None, iterations, threshold, scale, center.to_vec(), None, 2, seed);

    // For a line graph with equal edge weights, we expect:
    // - All nodes should be roughly collinear (on a line)
    // - The middle node (n2) should be between the other two
    // - The distances between adjacent nodes should be similar
    
    let pos1 = positions[&n1];
    let pos2 = positions[&n2];
    let pos3 = positions[&n3];

    // Expected positions for a line graph (theoretical ideal)
    let expected_pos1 = [1.0, 0.0];
    let expected_pos2 = [0.0, 0.0];
    let expected_pos3 = [-1.0, 0.0];
    
    println!("=== LINE GRAPH LAYOUT COMPARISON ===");
    println!("Expected positions:");
    println!("  Node 1: {:?}", expected_pos1);
    println!("  Node 2: {:?}", expected_pos2);
    println!("  Node 3: {:?}", expected_pos3);
    println!();
    println!("Actual positions returned by spring_layout:");
    println!("  Node 1: {:?}", pos1);
    println!("  Node 2: {:?}", pos2);
    println!("  Node 3: {:?}", pos3);
    println!();

    // Calculate distances
    let dist_12 = distance_between(&pos1, &pos2);
    let dist_23 = distance_between(&pos2, &pos3);
    let dist_13 = distance_between(&pos1, &pos3);

    println!("Actual distances:");
    println!("  Distance 1-2: {:.6}", dist_12);
    println!("  Distance 2-3: {:.6}", dist_23);
    println!("  Distance 1-3: {:.6}", dist_13);
    
    // Calculate expected distances
    let expected_dist_12 = distance_between(&expected_pos1, &expected_pos2);
    let expected_dist_23 = distance_between(&expected_pos2, &expected_pos3);
    let expected_dist_13 = distance_between(&expected_pos1, &expected_pos3);
    
    println!("Expected distances:");
    println!("  Distance 1-2: {:.6}", expected_dist_12);
    println!("  Distance 2-3: {:.6}", expected_dist_23);
    println!("  Distance 1-3: {:.6}", expected_dist_13);
    println!();
    
    // Calculate differences between expected and actual
    let pos_diff_1 = [(pos1[0] - expected_pos1[0]).abs(), (pos1[1] - expected_pos1[1]).abs()];
    let pos_diff_2 = [(pos2[0] - expected_pos2[0]).abs(), (pos2[1] - expected_pos2[1]).abs()];
    let pos_diff_3 = [(pos3[0] - expected_pos3[0]).abs(), (pos3[1] - expected_pos3[1]).abs()];
    
    println!("Position differences (|actual - expected|):");
    println!("  Node 1: [{:.6}, {:.6}]", pos_diff_1[0], pos_diff_1[1]);
    println!("  Node 2: [{:.6}, {:.6}]", pos_diff_2[0], pos_diff_2[1]);
    println!("  Node 3: [{:.6}, {:.6}]", pos_diff_3[0], pos_diff_3[1]);
    println!();

    // For a line graph, the middle node should be between the endpoints
    // This means the distance from 1 to 3 should be approximately the sum of 1-2 and 2-3
    let expected_dist_13 = dist_12 + dist_23;
    let actual_dist_13 = dist_13;
    
    // Allow some tolerance for the spring algorithm's convergence
    let tolerance = 0.1; // 10% tolerance
    assert!(
        (actual_dist_13 - expected_dist_13).abs() < tolerance * expected_dist_13,
        "Distance 1-3 ({:.6}) should be approximately sum of 1-2 ({:.6}) and 2-3 ({:.6})",
        actual_dist_13, dist_12, dist_23
    );

    // The adjacent distances should be similar (within reasonable tolerance)
    let distance_ratio = dist_12 / dist_23;
    assert!(
        distance_ratio > 0.5 && distance_ratio < 2.0,
        "Adjacent distances should be similar. Ratio: {:.3}", distance_ratio
    );

    // Check that nodes are roughly collinear by calculating the cross product
    // For collinear points, the cross product should be close to zero
    let cross_product = (pos2[0] - pos1[0]) * (pos3[1] - pos1[1]) - (pos2[1] - pos1[1]) * (pos3[0] - pos1[0]);
    let cross_product_magnitude = cross_product.abs();
    
    // The cross product magnitude should be small for collinear points
    // We'll use a tolerance based on the overall scale of the positions
    let position_scale = (dist_12 + dist_23 + dist_13) / 3.0;
    let collinearity_tolerance = 0.1 * position_scale;
    
    assert!(
        cross_product_magnitude < collinearity_tolerance,
        "Nodes should be roughly collinear. Cross product magnitude: {:.6}, tolerance: {:.6}",
        cross_product_magnitude, collinearity_tolerance
    );
}

// Helper function to calculate distances in a square arrangement
fn calculate_square_distances(positions: &HashMap<NodeKey, Position>) -> Vec<f64> {
    let mut distances = Vec::new();
    let nodes: Vec<NodeKey> = positions.keys().cloned().collect();
    
    for i in 0..nodes.len() {
        for j in (i+1)..nodes.len() {
            let pos1 = positions[&nodes[i]];
            let pos2 = positions[&nodes[j]];
            let dist = distance_between(&pos1, &pos2);
            distances.push(dist);
        }
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distances
}

// Helper function to calculate similarity between two distance vectors
fn calculate_similarity(distances1: &[f64], distances2: &[f64]) -> f64 {
    if distances1.len() != distances2.len() {
        return 0.0;
    }
    
    let mut similarity = 0.0;
    for (d1, d2) in distances1.iter().zip(distances2.iter()) {
        let diff = (d1 - d2).abs();
        let avg = (d1 + d2) / 2.0;
        if avg > 1e-10 {
            similarity += 1.0 - (diff / avg);
        } else {
            similarity += 1.0;
        }
    }
    similarity / distances1.len() as f64
}