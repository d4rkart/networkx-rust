use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::Debug;
use rand::Rng;
use rand::rngs::{ThreadRng, StdRng};
use rand::SeedableRng;
use rand::distributions::{Distribution, Uniform};

use crate::graph::{Graph, NodeKey};

/// Trait for extracting weights from edge data in a type-safe manner
pub trait WeightExtractor {
    /// Extract a weight value from the edge data using the specified field name
    /// Returns None if the weight field is not found or cannot be extracted
    fn extract_weight(&self, field: &str) -> Option<f64>;
}

impl WeightExtractor for f64 {
    fn extract_weight(&self, _f: &str) -> Option<f64> { Some(*self) }
}
impl WeightExtractor for i32 {
    fn extract_weight(&self, _f: &str) -> Option<f64> { Some(*self as f64) }
}
impl WeightExtractor for u32 {
    fn extract_weight(&self, _f: &str) -> Option<f64> { Some(*self as f64) }
}
impl WeightExtractor for f32 {
    fn extract_weight(&self, _f: &str) -> Option<f64> { Some(*self as f64) }
}
impl WeightExtractor for i64 {
    fn extract_weight(&self, _f: &str) -> Option<f64> { Some(*self as f64) }
}
impl WeightExtractor for u64 {
    fn extract_weight(&self, _f: &str) -> Option<f64> { Some(*self as f64) }
}

/// Implementation for &str - try JSON parsing or direct number parsing
impl WeightExtractor for &str {
    fn extract_weight(&self, field: &str) -> Option<f64> {
        // try JSON then plain number
        serde_json::from_str::<serde_json::Value>(self).ok()
            .and_then(|v| v.get(field).and_then(|x| x.as_f64()))
            .or_else(|| self.parse::<f64>().ok().map(|x| x as f64))
    }
}

/// Implementation for serde_json::Value - extract weight from JSON
impl WeightExtractor for serde_json::Value {
    fn extract_weight(&self, field: &str) -> Option<f64> {
        self.get(field).and_then(|v| v.as_f64())
    }
}

/// Implementation for unit type () - no weight
impl WeightExtractor for () {
    fn extract_weight(&self, _f: &str) -> Option<f64> { None }
}

/// Implementation for String - try JSON parsing or direct number parsing
impl WeightExtractor for String {
    fn extract_weight(&self, field: &str) -> Option<f64> {
        // Try to parse as JSON and extract the specified field
        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(self) {
            json_val.get(field)
                .and_then(|v| v.as_f64())
        } else {
            // Try to parse as plain number string
            self.parse::<f64>().ok()
        }
    }
}

pub type Position = [f64; 2];
pub type PositionMap = HashMap<NodeKey, Position>;

/// Rescale layout positions to (-scale, scale) in all dimensions.
/// 
/// To rescale, the mean (center) is subtracted from each axis separately.
/// Then all values are scaled so that the largest magnitude value
/// from all axes equals `scale` (thus, the aspect ratio is preserved).
fn rescale_layout(pos_array: &mut Vec<Vec<f64>>, scale: f64, center: &Vec<f64>) {
    let n_nodes = pos_array.len();
    let dim = if n_nodes > 0 { pos_array[0].len() } else { 0 };
    
    if n_nodes == 0 || dim == 0 {
        return;
    }

    // Compute mean per dimension
    let mut means = vec![0.0; dim];
    for p in &*pos_array { for d in 0..dim { means[d] += p[d]; } }
    for d in 0..dim { means[d] /= n_nodes as f64; }

    // Subtract mean and find global max abs across all dims
    let mut lim: f64 = 0.0;
    for i in 0..n_nodes {
        for d in 0..dim {
            pos_array[i][d] -= means[d];
            lim = lim.max(pos_array[i][d].abs());
        }
    }

    // Scale to (-scale, scale) and add center per dimension
    if lim > 0.0 {
        let factor = scale / lim;
        for i in 0..n_nodes {
            for d in 0..dim {
                pos_array[i][d] = pos_array[i][d] * factor + *center.get(d).unwrap_or(&0.0);
            }
        }
    } else {
        // If lim == 0, collapse at center
        for i in 0..n_nodes { 
            for d in 0..dim { 
                pos_array[i][d] = *center.get(d).unwrap_or(&0.0); 
            } 
        }
    }
}

/// Position nodes on concentric circles.
///
/// Parameters
/// ----------
/// G : Graph
///     A networkx graph
/// scale : f64 (default = 1.0)
///     Scale factor for positions
/// center : Position (default = [0.0, 0.0])
///     Center position for the layout
/// nlist : Vec<Vec<NodeKey>> (optional)
///     List of node lists for each shell
///
/// Returns
/// -------
/// PositionMap : HashMap<NodeKey, Position>
///     A mapping of nodes to positions
pub fn shell_layout<N, E>(
    graph: &Graph<N, E>,
    scale: f64,
    center: Position,
    nlist: Option<Vec<Vec<NodeKey>>>,
) -> PositionMap
where
    N: Clone + Debug,
    E: Clone + Debug,
{
    let mut pos = PositionMap::new();

    // If no nlist is provided, use all nodes in a single shell
    let shells = match nlist {
        Some(lists) => lists,
        None => {
            let nodes = graph.nodes();
            vec![nodes]
        }
    };

    // Check that all nodes are in some shell
    let mut shell_nodes = Vec::new();
    for shell in &shells {
        shell_nodes.extend(shell);
    }

    let all_nodes = graph.nodes();
    for node in all_nodes {
        if !shell_nodes.contains(&node) {
            panic!("Node {:?} not assigned to any shell", node);
        }
    }

    // Assign positions
    let mut radius = scale;

    for (i, shell) in shells.iter().enumerate() {
        let n_shell = shell.len();

        if n_shell == 0 {
            continue;
        }

        // Adjust the radius if needed
        if i == 0 && n_shell == 1 {
            radius = 0.0;
        }

        // Calculate positions for this shell
        for (j, &node) in shell.iter().enumerate() {
            let angle = 2.0 * PI * j as f64 / n_shell as f64;
            pos.insert(node, [
                center[0] + radius * angle.cos(),
                center[1] + radius * angle.sin()
            ]);
        }

        // Adjust radius for next shell
        radius += scale;
    }

    pos
}

/// Position nodes in a circular layout.
///
/// Parameters
/// ----------
/// G : Graph
///     A networkx graph
/// scale : f64 (default = 1.0)
///     Scale factor for positions
/// center : Position (default = [0.0, 0.0])
///     Center position for the layout
/// dim : usize (default = 2)
///     Dimension of layout (only 2D is supported)
///
/// Returns
/// -------
/// PositionMap : HashMap<NodeKey, Position>
///     A mapping of nodes to positions
pub fn circular_layout<N, E>(
    graph: &Graph<N, E>,
    scale: f64,
    center: Position,
) -> PositionMap
where
    N: Clone + Debug,
    E: Clone + Debug,
{
    let mut pos = PositionMap::new();

    let n_nodes = graph.node_count();
    if n_nodes == 0 {
        return pos;
    }

    if n_nodes == 1 {
        let only_node = graph.nodes()[0];
        pos.insert(only_node, center);
        return pos;
    }

    let radius = scale;
    let nodes = graph.nodes();

    for (i, &node) in nodes.iter().enumerate() {
        let theta = 2.0 * PI * i as f64 / n_nodes as f64;
        pos.insert(node, [
            center[0] + radius * theta.cos(),
            center[1] + radius * theta.sin()
        ]);
    }

    pos
}

/// Position nodes in a spiral layout.
///
/// Parameters
/// ----------
/// G : Graph
///     A networkx graph
/// scale : f64 (default = 1.0)
///     Scale factor for positions
/// center : Position (default = [0.0, 0.0])
///     Center position for the layout
/// resolution : f64 (default = 0.35)
///     The compactness of the spiral layout
/// equidistant : bool (default = false)
///     If true, nodes are placed at equal distances from each other.
///     If false, they're placed proportional to the integer order.
///
/// Returns
/// -------
/// PositionMap : HashMap<NodeKey, Position>
///     A mapping of nodes to positions
pub fn spiral_layout<N, E>(
    graph: &Graph<N, E>,
    scale: f64,
    center: Position,
    resolution: f64,
    equidistant: bool,
) -> PositionMap
where
    N: Clone + Debug,
    E: Clone + Debug,
{
    let mut pos = PositionMap::new();

    let n_nodes = graph.node_count();
    if n_nodes == 0 {
        return pos;
    }

    if n_nodes == 1 {
        let only_node = graph.nodes()[0];
        pos.insert(only_node, center);
        return pos;
    }

    let nodes = graph.nodes();

    for (i, &node) in nodes.iter().enumerate() {
        let i_float = i as f64;
        let theta = i_float * resolution;
        let r = if equidistant { scale * (theta / (2.0 * PI)).sqrt() } else { scale * (i_float / (n_nodes as f64 - 1.0)).sqrt() };

        pos.insert(node, [
            center[0] + r * theta.cos(),
            center[1] + r * theta.sin()
        ]);
    }

    pos
}

/// Position nodes using Fruchterman-Reingold force-directed algorithm.
///
/// Parameters
/// ----------
/// G : Graph
///     A networkx graph
/// k : Option<f64> (default = None)
///     Optimal distance between nodes. If None, the distance is set to
///     1.0 / sqrt(n) where n is the number of nodes
/// pos : Option<PositionMap> (default = None)
///     Initial positions for nodes
/// fixed : Option<&[NodeKey]> (default = None)
///     Nodes that should not be moved
/// iterations : usize (default = 50)
///     Maximum number of iterations
/// threshold : f64 (default = 1e-4)
///     Threshold for convergence
/// weight : Option<&str> (default = None)
///     The edge attribute that holds the numerical value used for the edge weight
/// scale : f64 (default = 1.0)
///     Scale factor for positions
/// center : Position (default = [0.0, 0.0])
///     Center position for the layout
/// dim : usize (default = 2)
///     Dimension of layout.
/// seed : Option<u64>
///     Used only for the initial positions in the algorithm.
///     Set the random state for deterministic node layouts
///
/// Returns
/// -------
/// PositionMap : HashMap<NodeKey, Position>
///     A mapping of nodes to positions
pub fn spring_layout<N, E>(
    graph: &Graph<N, E>,
    k: Option<f64>,
    pos: Option<&PositionMap>,
    fixed: Option<&[NodeKey]>,
    iterations: usize,
    threshold: f64,
    scale: f64,
    center: Vec<f64>,
    weight: Option<&str>,
    dim: usize,
    seed: Option<u64>,
) -> PositionMap
where
    N: Clone + Debug,
    E: Clone + Debug + WeightExtractor,
{
    let (dim, mut center) = if dim == 0 || dim == 1 { 
        (2, {
            let mut c = center;
            c.resize(2, 0.0);
            c
        })
    } else { 
        (dim, center) 
    };
    if center.len() != dim {
        panic!("spring_layout: center length ({}) must equal dim ({})", center.len(), dim);
    }
    let n_nodes = graph.node_count();

    if n_nodes == 0 {
        return HashMap::new();
    }

    if n_nodes == 1 {
        let mut positions = HashMap::new();
        let node = graph.nodes()[0];
        let cx = *center.get(0).unwrap_or(&0.0);
        let cy = *center.get(1).unwrap_or(&0.0);
        positions.insert(node, [cx, cy]);
        return positions;
    }

    // Create node list and index mapping
    let nodes = graph.nodes();
    let node_indices: HashMap<NodeKey, usize> = nodes.iter()
        .enumerate()
        .map(|(i, &node)| (node, i))
        .collect();

    // Set up positions
    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let mut pos_array: Vec<Vec<f64>> = Vec::with_capacity(n_nodes);

    if let Some(p) = pos {
        let mut dom_size: f64 = 0.0; // Max coord over all dims and positions
        for (&_node, coords) in p.iter() {
            for d in 0..dim {
                if d < coords.len() {
                    dom_size = dom_size.max(coords[d].abs());
                }
            }
        }
        if dom_size == 0.0 { dom_size = 1.0; }

        // Random initial positions scaled by dom_size, then add center
        for _ in 0..n_nodes {
            let mut coords = vec![0.0; dim];
            for d in 0..dim {
                coords[d] = rng.gen::<f64>() * dom_size;
                let c = *center.get(d).unwrap_or(&0.0);
                coords[d] += c;
            }
            pos_array.push(coords);
        }

        // Overwrite with provided positions
        for (i, &node) in nodes.iter().enumerate() {
            if let Some(provided_coords) = p.get(&node) {
                for d in 0..dim {
                    if d < provided_coords.len() {
                        pos_array[i][d] = provided_coords[d];
                    }
                }
            }
        }
    } else {
        // Generate random positions
        for _ in 0..n_nodes {
            let mut coords = vec![0.0; dim];
            for d in 0..dim {
                coords[d] = rng.gen::<f64>();
                let c = *center.get(d).unwrap_or(&0.0);
                coords[d] += c;
            }
            pos_array.push(coords);
        }
    }

    // Set up fixed nodes
    let fixed_nodes = if let Some(f) = fixed {
        f.iter().filter_map(|&node| {
            node_indices.get(&node).map(|&i| i)
        }).collect::<Vec<usize>>()
    } else {
        Vec::new()
    };

    // Optimal distance between nodes
    let k_val = k.unwrap_or_else(|| 1.0 / (n_nodes as f64).sqrt());

    // Create adjacency matrix
    let mut adj_matrix = vec![vec![0.0; n_nodes]; n_nodes];

    // Use the adjacency map directly for better performance
    let adjacency_map = graph.adj();

    for (u_idx, &node_u) in nodes.iter().enumerate() {
        // Get the edge data for this node from the graph's adjacency map
        if let Some(neighbors) = adjacency_map.get(&node_u) {
            for (&node_v, edge_data) in neighbors {
                let v_idx = node_indices[&node_v];

                // Extract the weight from the edge data using the specified weight field
                let weight = match weight {
                    // Use the trait-based approach for type-safe weight extraction
                    Some(field) => edge_data.extract_weight(field).unwrap_or(1.0),
                    // No weight field specified - use default weight of 1.0
                    None => edge_data.extract_weight("").unwrap_or(1.0),
                };

                adj_matrix[u_idx][v_idx] = weight;
            }
        }
    }

    // Calculate the initial temperature
    // This is about 0.1 of the domain area (1.0 x 1.0)
    let mut max_spread = 0.0;
    for d in 0..dim {
        let mut min_d = f64::INFINITY;
        let mut max_d = f64::NEG_INFINITY;
        for p in &pos_array {
            if d < p.len() {
                min_d = min_d.min(p[d]);
                max_d = max_d.max(p[d]);
            }
        }
        let spread_d = max_d - min_d;
        max_spread = f64::max(max_spread, spread_d);
    }
    let mut t = max_spread * 0.1;
    let dt = t / (iterations as f64 + 1.0);

    // Run the main loop
    for _ in 0..iterations {
        // Calculate displacement forces
        let mut disp: Vec<Vec<f64>> = vec![vec![0.0; dim]; n_nodes];

        // Calculate repulsive forces
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if i == j {
                    continue;
                }

                // Difference vector
                let mut distance_sq = 0.0;
                let mut dn = vec![0.0; dim];
                for d in 0..dim {
                    let diff = pos_array[i][d] - pos_array[j][d];
                    dn[d] = diff;
                    distance_sq += diff * diff;
                }
                let mut distance = distance_sq.sqrt();
                if distance < 0.01 {
                    distance = 0.01;
                }
                // Normalize difference vector
                for d in 0..dim { dn[d] /= distance; }

                // Repulsive force: k² / d
                let force = (k_val * k_val) / distance;

                for d in 0..dim { disp[i][d] += dn[d] * force; }
            }
        }

        // Calculate attractive forces
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if i == j || adj_matrix[i][j] == 0.0 {
                    continue;
                }

                // Difference vector
                let mut distance_sq = 0.0;
                let mut dn = vec![0.0; dim];
                for d in 0..dim {
                    let diff = pos_array[i][d] - pos_array[j][d];
                    dn[d] = diff;
                    distance_sq += diff * diff;
                }
                let mut distance = distance_sq.sqrt();
                if distance < 0.01 {
                    distance = 0.01;
                }

                // Normalized difference vector
                for d in 0..dim { dn[d] /= distance; }

                // Attractive force: d² / k * weight
                let force = distance * distance / k_val * adj_matrix[i][j];

                for d in 0..dim { disp[i][d] -= dn[d] * force; }
            }
        }

        let mut sum_delta_sq = 0.0;
        for i in 0..n_nodes {
            // Compute displacement length
            let mut length_sq = 0.0;
            for d in 0..dim { length_sq += disp[i][d] * disp[i][d]; }
            let mut length = length_sq.sqrt();
            if length < 0.01 { length = 0.01; }

            // Compute delta_pos
            let mut delta_vec = vec![0.0; dim];
            for d in 0..dim { delta_vec[d] = disp[i][d] * (t / length); }

            // Zero out fixed nodes movement
            if fixed_nodes.contains(&i) {
                for d in 0..dim { delta_vec[d] = 0.0; }
            }

            for d in 0..dim {
                pos_array[i][d] += delta_vec[d];
                sum_delta_sq += delta_vec[d] * delta_vec[d];
            }
        }

        // Cool the temperature
        t -= dt;

        // Check for convergence
        if (sum_delta_sq.sqrt() / n_nodes as f64) < threshold {
            break;
        }
    }

    // Rescale and center the layout
    if fixed.is_none() && scale != 0.0 {
        rescale_layout(&mut pos_array, scale, &center);
    }

    // Create the final position map
    let mut positions = HashMap::with_capacity(n_nodes);
    for (i, &node) in nodes.iter().enumerate() {
        // Project N-D to 2D for return type compatibility
        let x = pos_array[i][0];
        let y = pos_array[i][1];
        positions.insert(node, [x, y]);
    }

    positions
}

/// Helper function to generate random positions for nodes
pub fn random_layout<N, E>(
    graph: &Graph<N, E>,
    center: Position,
    scale: f64,
    rng: &mut ThreadRng,
) -> PositionMap
where
    N: Clone + Debug,
    E: Clone + Debug,
{
    let mut pos = PositionMap::new();
    let dist = Uniform::new(-scale, scale);

    let nodes = graph.nodes();
    for &node in &nodes {
        pos.insert(node, [
            center[0] + dist.sample(rng),
            center[1] + dist.sample(rng)
        ]);
    }

    pos
}

/// Position nodes using Kamada-Kawai path-length cost-function.
///
/// The algorithm minimizes the energy in a graph where springs are placed
/// between all pairs of nodes with spring lengths proportional to shortest
/// path lengths between nodes.
///
/// Parameters
/// ----------
/// G : Graph
///     A networkx graph
/// dist : Option<HashMap<(NodeKey, NodeKey), f64>> (default = None)
///     A two-level dictionary of optimal distances between nodes,
///     indexed by source and destination node.
///     If None, the distance is computed using shortest_path_length().
/// pos : Option<&PositionMap> (default = None)
///     Initial positions for nodes as a dictionary with node as keys
///     and values as a coordinate list or tuple.  If None, then use
///     circular_layout() for dim >= 2 and a linear layout for dim == 1.
/// weight : Option<&str> (default = None)
///     The edge attribute that holds the numerical value used for
///     the edge weight. If None, then all edge weights are 1.
/// scale : f64 (default = 1.0)
///     Scale factor for positions.
/// center : Position (default = [0.0, 0.0])
///     Coordinate pair around which to center the layout.
/// epsilon : f64 (default = 1e-5)
///     Maximum change in position for each iteration
/// max_iterations : usize (default = 1000)
///     Maximum number of iterations to run
///
/// Returns
/// -------
/// PositionMap : HashMap<NodeKey, Position>
///     A mapping of nodes to positions
pub fn kamada_kawai_layout<N, E>(
    graph: &Graph<N, E>,
    dist: Option<HashMap<(NodeKey, NodeKey), f64>>,
    pos: Option<&PositionMap>,
    _weight: Option<&str>,
    scale: f64,
    center: Position,
    epsilon: f64,
    max_iterations: usize,
) -> PositionMap
where
    N: Clone + Debug,
    E: Clone + Debug,
{
    let n_nodes = graph.node_count();

    if n_nodes == 0 {
        return HashMap::new();
    }

    if n_nodes == 1 {
        let mut positions = HashMap::new();
        let node = graph.nodes()[0];
        positions.insert(node, center);
        return positions;
    }

    // Get or compute distances between nodes
    let distances = match dist {
        Some(d) => d,
        None => {
            // Compute all shortest paths lengths
            compute_shortest_path_lengths(graph)
        }
    };

    // Initial layout - use circular layout if no positions provided
    let init_pos = match pos {
        Some(p) => p.clone(),
        None => circular_layout(graph, 1.0, [0.0, 0.0]),
    };

    // Create a complete graph with edges weighted by distance
    let mut pos_array: Vec<[f64; 2]> = Vec::with_capacity(n_nodes);
    let nodes = graph.nodes();

    // Initialize position array and compute L (ideal lengths)
    for &node in &nodes {
        let node_pos = *init_pos.get(&node).unwrap_or(&[0.0, 0.0]);
        pos_array.push(node_pos);
    }

    // Compute L and K matrices (ideal lengths and strengths)
    let mut l_matrix = vec![vec![0.0; n_nodes]; n_nodes];
    let mut k_matrix = vec![vec![0.0; n_nodes]; n_nodes];

    for (i, &node_i) in nodes.iter().enumerate() {
        for (j, &node_j) in nodes.iter().enumerate() {
            if i == j {
                continue;
            }

            let dist_ij = match distances.get(&(node_i, node_j)) {
                Some(&d) => d,
                None => {
                    // If no direct path exists, use a large value
                    let avg_dist = distances.values().sum::<f64>() / distances.len() as f64;
                    avg_dist * 10.0
                }
            };

            l_matrix[i][j] = dist_ij;
            k_matrix[i][j] = 1.0 / (dist_ij * dist_ij);
        }
    }

    // Main algorithm
    let delta_m = compute_delta(0, &pos_array, &k_matrix, &l_matrix, n_nodes);
    let mut max_delta = delta_m.norm();
    let mut node_m;

    let mut iteration = 0;
    while max_delta > epsilon && iteration < max_iterations {
        // Find node with max delta
        node_m = 0;
        max_delta = 0.0;

        for i in 0..n_nodes {
            let delta_i = compute_delta(i, &pos_array, &k_matrix, &l_matrix, n_nodes);
            let norm_i = delta_i.norm();

            if norm_i > max_delta {
                max_delta = norm_i;
                node_m = i;
            }
        }

        if max_delta < epsilon {
            break;
        }

        // Move node_m
        let mut new_pos_m = pos_array[node_m];
        let mut delta_m = compute_delta(node_m, &pos_array, &k_matrix, &l_matrix, n_nodes);

        // Local fix - Kamada-Kawai algorithm
        let mut inner_iterations = 0;
        let max_inner_iterations = 50;

        while delta_m.norm() > epsilon && inner_iterations < max_inner_iterations {
            // Compute hessian
            let hessian = compute_hessian(node_m, &pos_array, &k_matrix, &l_matrix, n_nodes);

            // Compute determinant and inverse
            let det = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0];
            if det.abs() < 1e-10 {
                break;
            }

            let inv_h = [
                [hessian[1][1] / det, -hessian[0][1] / det],
                [-hessian[1][0] / det, hessian[0][0] / det],
            ];

            // Compute new position
            new_pos_m[0] -= inv_h[0][0] * delta_m.dx + inv_h[0][1] * delta_m.dy;
            new_pos_m[1] -= inv_h[1][0] * delta_m.dx + inv_h[1][1] * delta_m.dy;

            // Update position and compute new delta
            pos_array[node_m] = new_pos_m;
            delta_m = compute_delta(node_m, &pos_array, &k_matrix, &l_matrix, n_nodes);

            inner_iterations += 1;
        }

        iteration += 1;
    }

    // Rescale and center layout
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for pos in &pos_array {
        min_x = min_x.min(pos[0]);
        max_x = max_x.max(pos[0]);
        min_y = min_y.min(pos[1]);
        max_y = max_y.max(pos[1]);
    }

    let width = max_x - min_x;
    let height = max_y - min_y;

    let size = width.max(height);
    if size > 0.0 {
        for pos in &mut pos_array {
            // Rescale to [0, 1]
            pos[0] = (pos[0] - min_x) / size;
            pos[1] = (pos[1] - min_y) / size;

            // Adjust scale and center
            pos[0] = center[0] + (pos[0] - 0.5) * scale * 2.0;
            pos[1] = center[1] + (pos[1] - 0.5) * scale * 2.0;
        }
    }

    // Create the final position map
    let mut positions = HashMap::with_capacity(n_nodes);
    for (i, &node) in nodes.iter().enumerate() {
        positions.insert(node, pos_array[i]);
    }

    positions
}

// Compute all shortest path lengths between nodes in the graph
fn compute_shortest_path_lengths<N, E>(graph: &Graph<N, E>) -> HashMap<(NodeKey, NodeKey), f64>
where
    N: Clone + Debug,
    E: Clone + Debug,
{
    let nodes = graph.nodes();
    let mut distances = HashMap::new();

    // For each pair of nodes, compute shortest path
    for &source in &nodes {
        // Currently we don't have Floyd-Warshall in the Graph, so we'll compute
        // each shortest path individually (less efficient)
        for &target in &nodes {
            if source == target {
                continue;
            }

            // Try to get shortest path - if none exists, we'll handle it in the main function
            if let Some((_, distance)) = graph.shortest_path(source, target, |_| 1.0) {
                distances.insert((source, target), distance);
            }
        }
    }

    distances
}

// Helper struct to represent a 2D gradient
struct Gradient {
    dx: f64,
    dy: f64,
}

impl Gradient {
    fn norm(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

// Compute the gradient (partial derivative) for a node
fn compute_delta(
    node_idx: usize,
    pos_array: &[[f64; 2]],
    k_matrix: &[Vec<f64>],
    l_matrix: &[Vec<f64>],
    n_nodes: usize,
) -> Gradient {
    let mut dx = 0.0;
    let mut dy = 0.0;

    for j in 0..n_nodes {
        if node_idx == j {
            continue;
        }

        let x_diff = pos_array[node_idx][0] - pos_array[j][0];
        let y_diff = pos_array[node_idx][1] - pos_array[j][1];

        let distance = (x_diff * x_diff + y_diff * y_diff).sqrt().max(1e-10);
        let k_ij = k_matrix[node_idx][j];
        let l_ij = l_matrix[node_idx][j];

        let force = k_ij * (1.0 - l_ij / distance);

        dx += force * x_diff;
        dy += force * y_diff;
    }

    Gradient { dx, dy }
}

// Compute the Hessian matrix (second derivatives) for a node
fn compute_hessian(
    node_idx: usize,
    pos_array: &[[f64; 2]],
    k_matrix: &[Vec<f64>],
    l_matrix: &[Vec<f64>],
    n_nodes: usize,
) -> [[f64; 2]; 2] {
    let mut hessian = [[0.0, 0.0], [0.0, 0.0]];

    for j in 0..n_nodes {
        if node_idx == j {
            continue;
        }

        let x_diff = pos_array[node_idx][0] - pos_array[j][0];
        let y_diff = pos_array[node_idx][1] - pos_array[j][1];

        let distance = (x_diff * x_diff + y_diff * y_diff).sqrt().max(1e-10);
        let k_ij = k_matrix[node_idx][j];
        let l_ij = l_matrix[node_idx][j];

        let force = k_ij * (1.0 - l_ij / distance);
        let force_distance = k_ij * l_ij / (distance * distance * distance);

        // Second derivatives
        hessian[0][0] += force + force_distance * x_diff * x_diff;
        hessian[0][1] += force_distance * x_diff * y_diff;
        hessian[1][0] += force_distance * y_diff * x_diff;
        hessian[1][1] += force + force_distance * y_diff * y_diff;
    }

    hessian
}