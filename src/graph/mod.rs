// Export the graph module
pub mod graph;
pub mod digraph;
pub mod multigraph;
pub mod multidigraph;

// Re-export commonly used types from the graph module
pub use graph::{Graph, NodeKey};
pub use digraph::DiGraph;
pub use multigraph::MultiGraph;
pub use multidigraph::MultiDiGraph;