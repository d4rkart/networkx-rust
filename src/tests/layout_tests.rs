use crate::{Graph, Position, circular_layout, shell_layout, spiral_layout, spring_layout, kamada_kawai_layout, random_layout};
use rand::thread_rng;

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

    let positions = spring_layout(&graph, k, None, None, iterations, threshold, scale, center, None);

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
