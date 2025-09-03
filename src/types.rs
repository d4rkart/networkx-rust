use std::collections::HashMap;
use std::any::Any;

pub trait Cloneable: Any {
    fn clone_box(&self) -> Box<dyn Cloneable>;
}

impl<T: Any + Clone> Cloneable for T {
    fn clone_box(&self) -> Box<dyn Cloneable> {
        Box::new(self.clone())
    }
}

pub type EdgeAttr = HashMap<String, Box<dyn Cloneable>>;
pub type NodeAttr = HashMap<String, Box<dyn Cloneable>>;
pub type GraphAttr = HashMap<String, Box<dyn Cloneable>>;

pub fn clone_node_attr(attr: &NodeAttr) -> NodeAttr {
    attr.iter().map(|(k, v)| (k.clone(), v.clone_box())).collect()
}

pub fn clone_edge_attr(attr: &EdgeAttr) -> EdgeAttr {
    attr.iter().map(|(k, v)| (k.clone(), v.clone_box())).collect()
}

pub fn clone_graph_attr(attr: &GraphAttr) -> GraphAttr {
    attr.iter().map(|(k, v)| (k.clone(), v.clone_box())).collect()
}