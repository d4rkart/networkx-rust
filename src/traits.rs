use std::hash::Hash;
use crate::types::{EdgeAttr, NodeAttr, GraphAttr};
use std::any::Any;
pub trait GraphLike<T: Hash + Eq + Clone> {
    fn nodes_with_data(&self) -> Vec<(T, NodeAttr)>;
    fn edges_with_data(&self) -> Vec<(T, T, EdgeAttr)>;
    fn graph_attrs(&self) -> Vec<(String, Box<dyn Any>)>;
}