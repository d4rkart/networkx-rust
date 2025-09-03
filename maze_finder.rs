use networkx_rs::{Graph, NodeKey};
use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// Function to create a grid-based maze
fn create_maze(width: usize, height: usize, walls: &[(usize, usize)]) -> Graph<(usize, usize), u32> {
    let mut maze = Graph::with_name(false, "Maze");
    
    // Create a mapping from coordinates to node keys for faster lookup
    let mut coord_to_key = HashMap::new();
    
    // Add all non-wall cells as nodes in the graph
    for y in 0..height {
        for x in 0..width {
            if !walls.contains(&(x, y)) {
                let key = maze.add_node((x, y));
                coord_to_key.insert((x, y), key);
            }
        }
    }
    
    // Add edges between adjacent non-wall cells (up, down, left, right)
    for y in 0..height {
        for x in 0..width {
            if let Some(&from_key) = coord_to_key.get(&(x, y)) {
                // Check all four adjacent cells
                for (dx, dy) in [(0, 1), (1, 0), (0, -1), (-1, 0)].iter() {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    
                    // Skip if the adjacent cell is out of bounds
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    
                    let nx = nx as usize;
                    let ny = ny as usize;
                    
                    // Add an edge if the adjacent cell is not a wall
                    if let Some(&to_key) = coord_to_key.get(&(nx, ny)) {
                        maze.add_edge(from_key, to_key, 1); // Weight of 1 for each step
                    }
                }
            }
        }
    }
    
    maze
}

// Function to find the shortest path in the maze
fn find_path(
    maze: &Graph<(usize, usize), u32>,
    start: (usize, usize),
    end: (usize, usize),
    coord_to_key: &HashMap<(usize, usize), NodeKey>
) -> Option<(Vec<(usize, usize)>, u32)> {
    // Get node keys for start and end positions
    let start_key = *coord_to_key.get(&start)?;
    let end_key = *coord_to_key.get(&end)?;
    
    // Find the shortest path using the graph's built-in function
    let (path_keys, cost) = maze.shortest_path(start_key, end_key, |&w| w as f64)?;
    
    // Convert node keys back to coordinates
    let path_coords = path_keys.iter()
        .map(|&key| *maze.get_node_data(key).unwrap())
        .collect();
    
    Some((path_coords, cost as u32))
}

// Function to print the maze with the path
fn print_maze(
    width: usize,
    height: usize,
    walls: &[(usize, usize)],
    path: &[(usize, usize)],
    start: (usize, usize),
    end: (usize, usize)
) {
    for y in 0..height {
        for x in 0..width {
            let pos = (x, y);
            
            if pos == start {
                print!("S ");
            } else if pos == end {
                print!("E ");
            } else if walls.contains(&pos) {
                print!("# ");
            } else if path.contains(&pos) {
                print!("* ");
            } else {
                print!(". ");
            }
        }
        println!();
    }
}

// Function to generate a maze with dynamic walls
fn generate_dynamic_maze(width: usize, height: usize, wall_density: f64, seed: Option<u64>) -> Vec<(usize, usize)> {
    let mut walls = Vec::new();
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    
    // Generate random walls based on density
    for y in 0..height {
        for x in 0..width {
            // Avoid placing walls at start (0,0) and end (width-1, height-1)
            if (x == 0 && y == 0) || (x == width - 1 && y == height - 1) {
                continue;
            }
            
            if rng.gen::<f64>() < wall_density {
                walls.push((x, y));
            }
        }
    }
    
    // Add some structure to the maze with partial walls
    let num_obstacles = (width.min(height) / 5).max(1);
    
    // Add some vertical walls
    for _ in 0..num_obstacles {
        let wall_x = rng.gen_range(width / 4..3 * width / 4);
        let wall_length = rng.gen_range(height / 3..2 * height / 3);
        let start_y = rng.gen_range(0..height - wall_length);
        
        // Leave a gap in the wall
        let gap_pos = rng.gen_range(0..wall_length);
        
        for i in 0..wall_length {
            if i != gap_pos {
                walls.push((wall_x, start_y + i));
            }
        }
    }
    
    // Add some horizontal walls
    for _ in 0..num_obstacles {
        let wall_y = rng.gen_range(height / 4..3 * height / 4);
        let wall_length = rng.gen_range(width / 3..2 * width / 3);
        let start_x = rng.gen_range(0..width - wall_length);
        
        // Leave a gap in the wall
        let gap_pos = rng.gen_range(0..wall_length);
        
        for i in 0..wall_length {
            if i != gap_pos {
                walls.push((start_x + i, wall_y));
            }
        }
    }
    
    walls
}

fn main() {
    // Allow configurable maze dimensions
    let args: Vec<String> = std::env::args().collect();
    
    // Default values
    let mut width = 500;
    let mut height = 500;
    let mut wall_density = 0.2;
    let mut seed: Option<u64> = None;
    
    // Parse command line arguments if provided
    if args.len() > 1 {
        width = args[1].parse().unwrap_or(50);
    }
    if args.len() > 2 {
        height = args[2].parse().unwrap_or(50);
    }
    if args.len() > 3 {
        wall_density = args[3].parse::<f64>().unwrap_or(0.2);
        // Clamp wall density between 0.1 and 0.5
        wall_density = wall_density.max(0.1).min(0.5);
    }
    if args.len() > 4 {
        seed = Some(args[4].parse().unwrap_or(42));
    }
    
    println!("Generating maze of size {}x{} with wall density {}", width, height, wall_density);
    
    // Generate walls dynamically
    let walls = generate_dynamic_maze(width, height, wall_density, seed);
    
    // Define start and end positions
    let start = (0, 0);
    let end = (width - 1, height - 1);
    
    println!("Finding path from {:?} to {:?}", start, end);
    
    // Create the maze graph
    let maze = create_maze(width, height, &walls);
    
    // Create a mapping from coordinates to node keys
    let mut coord_to_key = HashMap::new();
    for key in maze.nodes() {
        let coord = *maze.get_node_data(key).unwrap();
        coord_to_key.insert(coord, key);
    }
    
    // Find the shortest path
    match find_path(&maze, start, end, &coord_to_key) {
        Some((path, cost)) => {
            println!("Found path of length {} with {} steps!", path.len(), cost);
            
            // For larger mazes, only print if it's reasonable size
            if width <= 100 && height <= 100 {
                print_maze(width, height, &walls, &path, start, end);
            } else {
                println!("Maze too large to print. Path found successfully!");
            }
        },
        None => println!("No path found from {:?} to {:?}!", start, end),
    }
} 