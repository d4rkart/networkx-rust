use networkx_rs::Graph;
use std::collections::HashMap;

// A more complex pathfinding example
fn main() {
    // Create a subway network graph
    #[derive(Debug, Clone)]
    struct Station {
        name: String,
        line: String,
        has_transfer: bool,
    }

    #[derive(Debug, Clone)]
    struct Connection {
        distance: f64, // in km
        travel_time: f64, // in minutes
    }

    let mut subway = Graph::<Station, Connection>::new(false);

    // Add stations for Line 1 (Red Line)
    let red_line_stations = [
        ("Central", true),
        ("Riverfront", false),
        ("Market", true),
        ("Downtown", true),
        ("Parkside", false),
        ("University", true),
        ("Tech District", false),
        ("Airport", true),
    ];

    // Add stations for Line 2 (Blue Line)
    let blue_line_stations = [
        ("Harbor", false),
        ("Beachfront", false),
        ("Market", true),
        ("City Hall", false),
        ("Stadium", true),
        ("Hospital", false),
        ("University", true),
    ];

    // Add stations for Line 3 (Green Line)
    let green_line_stations = [
        ("Suburbs", false),
        ("Shopping Center", false),
        ("Downtown", true),
        ("Financial District", false),
        ("Stadium", true),
        ("Convention Center", false),
        ("Airport", true),
    ];

    // Add stations to the graph
    let mut stations = HashMap::new();

    // Add Red Line stations
    for (i, (name, has_transfer)) in red_line_stations.iter().enumerate() {
        let station = Station {
            name: name.to_string(),
            line: "Red Line".to_string(),
            has_transfer: *has_transfer,
        };
        let key = subway.add_node(station);
        stations.insert(format!("Red-{}", name), key);

        // Connect to previous station
        if i > 0 {
            let prev_name = red_line_stations[i-1].0;
            let prev_key = stations[&format!("Red-{}", prev_name)];
            
            // Random but realistic distance and time between stations
            let distance = 1.0 + (i % 3) as f64 * 0.5; // 1.0 to 2.0 km
            let travel_time = 2.0 + distance * 1.5; // base time + distance factor
            
            subway.add_edge(prev_key, key, Connection { 
                distance, 
                travel_time,
            });
        }
    }

    // Add Blue Line stations
    for (i, (name, has_transfer)) in blue_line_stations.iter().enumerate() {
        // If this station exists on another line as a transfer point, don't add it again
        if *has_transfer && stations.contains_key(&format!("Red-{}", name)) {
            // Just store a reference to the existing station
            stations.insert(format!("Blue-{}", name), stations[&format!("Red-{}", name)]);
        } else {
            let station = Station {
                name: name.to_string(),
                line: "Blue Line".to_string(),
                has_transfer: *has_transfer,
            };
            let key = subway.add_node(station);
            stations.insert(format!("Blue-{}", name), key);
        }

        // Connect to previous station
        if i > 0 {
            let prev_name = blue_line_stations[i-1].0;
            let prev_key = stations[&format!("Blue-{}", prev_name)];
            let curr_key = stations[&format!("Blue-{}", name)];
            
            // Only add an edge if these are different stations
            if prev_key != curr_key {
                let distance = 1.0 + (i % 4) as f64 * 0.4; // 1.0 to 2.2 km
                let travel_time = 2.0 + distance * 1.5; // base time + distance factor
                
                subway.add_edge(prev_key, curr_key, Connection { 
                    distance, 
                    travel_time,
                });
            }
        }
    }

    // Add Green Line stations
    for (i, (name, has_transfer)) in green_line_stations.iter().enumerate() {
        // Check if this station exists on another line as a transfer point
        if *has_transfer {
            if stations.contains_key(&format!("Red-{}", name)) {
                stations.insert(format!("Green-{}", name), stations[&format!("Red-{}", name)]);
            } else if stations.contains_key(&format!("Blue-{}", name)) {
                stations.insert(format!("Green-{}", name), stations[&format!("Blue-{}", name)]);
            } else {
                let station = Station {
                    name: name.to_string(),
                    line: "Green Line".to_string(),
                    has_transfer: true,
                };
                let key = subway.add_node(station);
                stations.insert(format!("Green-{}", name), key);
            }
        } else {
            let station = Station {
                name: name.to_string(),
                line: "Green Line".to_string(),
                has_transfer: false,
            };
            let key = subway.add_node(station);
            stations.insert(format!("Green-{}", name), key);
        }

        // Connect to previous station
        if i > 0 {
            let prev_name = green_line_stations[i-1].0;
            let prev_key = stations[&format!("Green-{}", prev_name)];
            let curr_key = stations[&format!("Green-{}", name)];
            
            // Only add an edge if these are different stations
            if prev_key != curr_key {
                let distance = 1.0 + (i % 3) as f64 * 0.6; // 1.0 to 2.2 km
                let travel_time = 2.0 + distance * 1.5; // base time + distance factor
                
                subway.add_edge(prev_key, curr_key, Connection { 
                    distance, 
                    travel_time,
                });
            }
        }
    }

    println!("Subway Network Information:");
    println!("Number of stations: {}", subway.node_count());
    println!("Number of connections: {}", subway.edge_count());
    
    // Find the shortest path by distance
    let start_station = stations[&"Red-Central".to_string()];
    let end_station = stations[&"Green-Airport".to_string()];
    
    println!("\nFinding shortest path by distance from Central to Airport:");
    
    let (path, total_distance) = subway.shortest_path(
        start_station, 
        end_station, 
        |conn| conn.distance
    ).unwrap();
    
    println!("Total distance: {:.1} km", total_distance);
    print_path(&subway, &path, |conn| format!("{:.1} km", conn.distance));
    
    // Find the shortest path by travel time
    println!("\nFinding shortest path by travel time from Central to Airport:");
    
    let (path, total_time) = subway.shortest_path(
        start_station, 
        end_station, 
        |conn| conn.travel_time
    ).unwrap();
    
    println!("Total travel time: {:.1} minutes", total_time);
    print_path(&subway, &path, |conn| format!("{:.1} min", conn.travel_time));
    
    // Find path between other stations
    let start_station = stations[&"Blue-Harbor".to_string()];
    let end_station = stations[&"Red-Tech District".to_string()];
    
    println!("\nFinding shortest path by travel time from Harbor to Tech District:");
    
    match subway.shortest_path(
        start_station, 
        end_station, 
        |conn| conn.travel_time
    ) {
        Some((path, total_time)) => {
            println!("Total travel time: {:.1} minutes", total_time);
            print_path(&subway, &path, |conn| format!("{:.1} min", conn.travel_time));
        },
        None => println!("No path found!"),
    }
}

// Helper function to print the path with station names and connection details
fn print_path<T, E, F>(
    graph: &Graph<T, E>, 
    path: &[usize],
    format_edge: F
) 
where 
    T: Clone + std::fmt::Debug,
    E: Clone + std::fmt::Debug,
    F: Fn(&E) -> String,
{
    println!("Path:");
    for (i, &station_key) in path.iter().enumerate() {
        let station = graph.get_node_data(station_key).unwrap();
        print!("{:?}", station);
        
        if i < path.len() - 1 {
            let next_key = path[i + 1];
            let connection = graph.get_edge_data(station_key, next_key).unwrap();
            print!(" --({})-> ", format_edge(connection));
        }
        
        if i < path.len() - 1 {
            println!();
        }
    }
    println!();
} 