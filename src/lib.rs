//! Utilities for manipulating tree graphs, for the analysis of neuronal arbors.
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

pub use slab_tree;
use slab_tree::{NodeId, NodeRef, RemoveBehavior, Tree, TreeBuilder};

pub type Precision = f64;

/// Trait adding some topological utilities to a tree representation.
pub trait TopoArbor {
    type Node;

    /// Remove the given nodes and everything below them.
    /// Some nodes in the starting set may have been removed as
    /// descendants of others.
    fn prune_at(&mut self, node_ids: &[NodeId]) -> HashSet<NodeId>;

    /// Remove everything distal to the given node,
    /// and up to the next branch proximal of it.
    fn prune_branches_containing(&mut self, node_ids: &[NodeId]) -> HashSet<NodeId>;

    /// Remove all branches with a strahler index less than `threshold`.
    fn prune_below_strahler(&mut self, threshold: usize) -> HashSet<NodeId>;

    /// Remove all branches greater than `threshold` branch points from the root.
    fn prune_beyond_branches(&mut self, threshold: usize) -> HashSet<NodeId>;

    /// Remove all nodes greater than `threshold` steps from the root.
    fn prune_beyond_steps(&mut self, threshold: usize) -> HashSet<NodeId>;

    // TODO: iterator?
    /// Decompose the arbor into slabs: unbranched runs of nodes.
    /// The start of every slab is the root or a branch point,
    /// and the end of every slab is a branch point or leaf.
    /// Returned depth first in preorder.
    fn slabs(&self) -> Vec<Vec<NodeId>>;

    // fn get_node(&self, node_id: NodeId) -> Option<NodeRef<Self::Node>>;
}

/// Given tuples of (child_id, optional_parent_id, child_data),
/// make a tree whose node data are (id, data).
/// Returns that tree, and a mapping from the passed-in IDs to the internal IDs.
pub fn edges_to_tree_with_data<T: Hash + Eq + Copy, D: Clone>(
    edges: &[(T, Option<T>, D)],
) -> Result<(Tree<(T, D)>, HashMap<T, NodeId>), &'static str> {
    let size = edges.len();
    let mut root_opt: Option<T> = None;
    let mut data: HashMap<T, D> = HashMap::with_capacity(size);
    let mut child_vecs: HashMap<T, Vec<T>> = HashMap::with_capacity(size);

    for (child, parent_opt, d) in edges.iter() {
        data.insert(*child, d.clone());
        match parent_opt {
            Some(p) => child_vecs
                .entry(*p)
                .or_insert_with(Vec::default)
                .push(*child),
            None => {
                if root_opt.is_some() {
                    return Err("More than one root");
                }
                root_opt.replace(*child);
            }
        }
    }

    let root_tnid = root_opt.ok_or("No root")?;
    let mut tree = TreeBuilder::new()
        .with_capacity(edges.len())
        .with_root((root_tnid, data.remove(&root_tnid).unwrap()))
        .build();

    let mut tnid_to_id = HashMap::default();
    tnid_to_id.insert(root_tnid, tree.root_id().unwrap());

    // ? can we use the NodeMut object here? lifetime issues
    let mut to_visit = vec![tree.root_id().expect("Just set root")];
    while let Some(node_id) = to_visit.pop() {
        let mut parent = tree.get_mut(node_id).expect("Just placed");
        let parent_data = &parent.data();
        if let Some(v) = child_vecs.remove(&parent_data.0) {
            to_visit.extend(
                v.into_iter()
                    .map(|tnid| {
                        let datum = data.remove(&tnid).unwrap();
                        let node_id = parent.append((tnid, datum)).node_id();
                        tnid_to_id.insert(tnid, node_id);
                        node_id
                    })
            );
        }
    }

    Ok((tree, tnid_to_id))
}


impl<T: Debug> TopoArbor for Tree<T> {
    type Node = T;

    fn prune_at(&mut self, node_ids: &[NodeId]) -> HashSet<NodeId> {
        let mut pruned = HashSet::with_capacity(node_ids.len());
        for node_id in node_ids {
            if self
                .remove(*node_id, RemoveBehavior::DropChildren)
                .is_some()
            {
                pruned.insert(*node_id);
            }
        }

        pruned
    }

    fn prune_branches_containing(&mut self, node_ids: &[NodeId]) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut to_remove = Vec::default();
        for node_id in node_ids {
            let mut ancestor = *node_id;
            while let Some(node) = self.get(ancestor) {
                // seem to be going id -> node -> id more than necessary?
                if visited.contains(&ancestor) {
                    break;
                } else if node.prev_sibling().is_some() || node.next_sibling().is_some() {
                    to_remove.push(ancestor);
                    break;
                }

                visited.insert(ancestor);
                ancestor = match node.parent() {
                    Some(n) => n.node_id(),
                    _ => break,
                };
            }
        }
        self.prune_at(&to_remove)
    }

    fn prune_below_strahler(&mut self, threshold: usize) -> HashSet<NodeId> {
        let mut strahler: HashMap<NodeId, usize> = HashMap::default();
        let mut to_prune = Vec::default();
        for node in self.root().expect("must have a root").traverse_post_order() {
            let mut max_child_strahler = 0;
            let mut max_strahler_count = 0;
            let mut sub_threshold = Vec::default();
            for child in node.children() {
                let child_strahler = strahler
                    .remove(&child.node_id())
                    .expect("If it has children, they must have been visited");
                if child_strahler < threshold {
                    sub_threshold.push(child.node_id());
                }
                match child_strahler.cmp(&max_child_strahler) {
                    Ordering::Greater => {
                        max_child_strahler = child_strahler;
                        max_strahler_count = 1;
                    }
                    Ordering::Equal => max_strahler_count += 1,
                    _ => (),
                }
            }
            let node_strahler = match max_strahler_count.cmp(&1) {
                Ordering::Equal => max_child_strahler,
                Ordering::Greater => max_child_strahler + 1,
                _ => 1,
            };
            if node_strahler == threshold {
                to_prune.extend(sub_threshold.into_iter());
            }
            strahler.insert(node.node_id(), node_strahler);
        }
        self.prune_at(&to_prune)
    }

    fn prune_beyond_branches(&mut self, threshold: usize) -> HashSet<NodeId> {
        let mut to_prune = Vec::default();
        let mut to_visit = vec![(self.root().expect("must have root"), 0)];
        while let Some((node, level)) = to_visit.pop() {
            let children: Vec<NodeRef<T>> = node.children().collect();
            if children.len() > 1 {
                if level >= threshold {
                    to_prune.extend(children.into_iter().map(|n| n.node_id()));
                } else {
                    to_visit.extend(children.into_iter().map(|n| (n, level + 1)));
                }
            } else {
                to_visit.extend(children.into_iter().map(|n| (n, level)));
            }
        }
        self.prune_at(&to_prune)
    }

    fn prune_beyond_steps(&mut self, threshold: usize) -> HashSet<NodeId> {
        let mut to_prune = Vec::default();
        let mut to_visit = vec![(self.root().expect("must have root"), 0)];
        while let Some((node, steps)) = to_visit.pop() {
            if steps >= threshold {
                to_prune.extend(node.children().map(|n| n.node_id()));
            } else {
                let new_steps = steps + 1;
                to_visit.extend(node.children().map(|n| (n, new_steps)));
            }
        }
        self.prune_at(&to_prune)
    }

    fn slabs(&self) -> Vec<Vec<NodeId>> {
        let mut to_visit = vec![vec![self.root().expect("must have root").node_id()]];
        let mut slabs = Vec::default();
        while let Some(mut slab) = to_visit.pop() {
            let mut tail = self
                .get(*slab.last().expect("has length"))
                .expect("has node");
            loop {
                let mut children: Vec<NodeRef<T>> = tail.children().collect();
                match children.len().cmp(&1) {
                    Ordering::Greater => {
                        to_visit.extend(
                            children
                                .into_iter()
                                .map(|c| vec![tail.node_id(), c.node_id()]),
                        );
                        break;
                    }
                    Ordering::Equal => {
                        tail = children.pop().expect("know it exists");
                        slab.push(tail.node_id());
                    }
                    Ordering::Less => break,
                }
            }
            slabs.push(slab);
        }
        slabs
    }
}

// ? generic so that different Locations can be cross-compared
// Trait for types which describe a 3D point.
pub trait Location {
    /// Where the point is, in 3D space
    fn location(&self) -> &[Precision; 3];

    /// How far from one Location object to another
    fn distance_to(&self, other: &[Precision; 3]) -> Precision {
        let mut squares_total = 0.0;
        for (a, b) in self.location().iter().zip(other.location().iter()) {
            squares_total += (a - b).powf(2.0);
        }
        squares_total.sqrt()
    }

    /// Where you would end up if you travelled `distance` towards `other`,
    /// and the overshoot: how far past the point you have travelled
    /// (negative if the point was not reached).
    fn project_towards(
        &self,
        other: &[Precision; 3],
        distance: Precision,
    ) -> ([Precision; 3], Precision) {
        let self_loc = self.location();
        let distance_to = self.distance_to(other);
        if distance_to * distance == 0.0 {
            return (*self_loc, 0.0);
        }
        let mut out = [0.0, 0.0, 0.0];
        for (idx, (a, b)) in self_loc.iter().zip(other.location().iter()).enumerate() {
            let diff = b - a;
            out[idx] = a + (diff / distance_to) * distance;
        }
        (out, distance - distance_to)
    }
}

impl Location for [Precision; 3] {
    fn location(&self) -> &[Precision; 3] {
        self
    }
}

impl Location for &[Precision; 3] {
    fn location(&self) -> &[Precision; 3] {
        self
    }
}

impl<Id, L: Location> Location for (Id, L) {
    fn location(&self) -> &[Precision; 3] {
        self.1.location()
    }
}

impl<Id, L: Location> Location for &(Id, L) {
    fn location(&self) -> &[Precision; 3] {
        self.1.location()
    }
}

// TODO: take iterator, return iterator
/// Place one point at the start of the linestring.
/// Travel down the linestring, placing another point at intervals of `length`,
/// until you reach the end.
/// Return all the placed points.
pub fn resample_linestring(linestring: &[impl Location], length: Precision) -> Vec<[Precision; 3]> {
    if length <= 0.0 {
        // TODO: result
        panic!("Can't resample with length <= 0");
    }
    let mut it = linestring.iter().map(|s| s.location());
    let mut prev = match it.next() {
        Some(p) => *p,
        _ => return vec![],
    };
    let mut out = vec![prev];
    let mut remaining = length;

    let mut next_opt = it.next();
    while let Some(next) = next_opt {
        if remaining <= 0.0 {
            remaining = length
        }
        let (new, overshoot) = prev.project_towards(next, remaining);
        match overshoot.partial_cmp(&0.0).expect("Non-numeric float") {
            Ordering::Greater => {
                // we've overshot
                remaining = overshoot;
                next_opt = it.next();
                prev = *next;
            }
            Ordering::Less => {
                // we've undershot (overshoot is negative)
                remaining = length;
                out.push(new);
                prev = new;
            }
            Ordering::Equal => {
                remaining = length;
                out.push(new);
                prev = new;
                next_opt = it.next();
            }
        };
    }
    out
}

/// Keeps root, branches, and leaves: otherwise, resample each slab with the given length.
pub fn resample_tree_points<T: Location + Debug>(
    tree: Tree<T>,
    length: Precision,
) -> Vec<[Precision; 3]> {
    let id_slabs = tree.slabs();
    let root_loc = tree.get(id_slabs[0][0]).unwrap().data().location();
    let mut out = vec![*root_loc];

    for slab_ids in id_slabs.into_iter() {
        let slab_locs: Vec<_> = slab_ids
            .into_iter()
            .map(|sid| tree.get(sid).unwrap().data().location())
            .collect();
        out.extend(resample_linestring(&slab_locs, length).into_iter().skip(1));
        out.push(**slab_locs.last().unwrap());
    }

    out
}

#[cfg(test)]
mod tests {
    use crate::*;
    use std::fmt::Debug;

    const EPSILON: Precision = 0.0001;

    /// From [wikipedia](https://en.wikipedia.org/wiki/Tree_traversal#/media/File:Sorted_binary_tree_ALL.svg)
    ///
    ///     F
    ///    / \
    ///   B   G
    ///  / \   \
    /// A   D   I
    ///    / \   \
    ///   C   E   H
    fn make_topotree() -> (Tree<&'static str>, HashMap<&'static str, NodeId>) {
        let mut tree = TreeBuilder::new().with_capacity(9).with_root("F").build();
        let mut f = tree.root_mut().unwrap();
        let mut b = f.append("B");
        b.append("A");
        let mut d = b.append("D");
        d.append("C");
        d.append("E");
        f.append("G").append("I").append("H");

        let map = f
            .as_ref()
            .traverse_pre_order()
            .map(|n| (*n.data(), n.node_id()))
            .collect();
        print_tree(&tree, "ORIGINAL");

        (tree, map)
    }

    fn nodes<T: Hash + Eq + Copy>(tree: &Tree<T>) -> HashSet<T> {
        tree.root()
            .unwrap()
            .traverse_pre_order()
            .map(|n| *n.data())
            .collect()
    }

    fn print_tree<T: Debug>(tree: &Tree<T>, label: &'static str) {
        let mut s = String::new();
        tree.write_formatted(&mut s).unwrap();
        println!("{}\n{}", label, s);
    }

    fn assert_nodes<T: Debug + Hash + Eq + Copy>(
        tree: &Tree<T>,
        contains: &[T],
        not_contains: &[T],
    ) {
        print_tree(tree, "RESULT");
        let tns = nodes(tree);
        for n in contains {
            assert!(tns.contains(n));
        }
        for n in not_contains {
            assert!(!tns.contains(n));
        }
    }

    #[test]
    fn prune_at() {
        let (mut tree, map) = make_topotree();
        tree.prune_at(&[map["G"]]);
        assert_nodes(&tree, &["F"], &["G", "H", "I"]);
    }

    #[test]
    fn prune_containing() {
        let (mut tree, map) = make_topotree();
        tree.prune_branches_containing(&[map["I"]]);
        assert_nodes(&tree, &["F"], &["G", "H", "I"]);
    }

    #[test]
    fn prune_below_strahler() {
        let (mut tree, _) = make_topotree();
        tree.prune_below_strahler(2);
        assert_nodes(&tree, &["F", "B", "D"], &["C", "E", "G"]);
    }

    #[test]
    fn prune_beyond_branches() {
        let (mut tree, _) = make_topotree();
        tree.prune_beyond_branches(2);
        assert_nodes(&tree, &["D", "A", "H"], &["C", "E"]);
    }

    #[test]
    fn prune_beyond_steps() {
        let (mut tree, _) = make_topotree();
        tree.prune_beyond_steps(1);
        assert_nodes(&tree, &["B", "G"], &["A", "D", "I"]);
    }

    fn add_points(a: &[Precision; 3], b: &[Precision; 3]) -> [Precision; 3] {
        let mut v = Vec::with_capacity(3);
        for (this_a, this_b) in a.iter().zip(b.iter()) {
            v.push(this_a + this_b);
        }
        [v[0], v[1], v[2]]
    }

    fn make_linestring(
        start: &[Precision; 3],
        step: &[Precision; 3],
        count: usize,
    ) -> Vec<[Precision; 3]> {
        let mut out = vec![*start];
        for _ in 0..(count - 1) {
            let next = add_points(out.last().unwrap(), step);
            out.push(next);
        }
        out
    }

    fn assert_close(a: Precision, b: Precision) {
        if (a - b).abs() >= EPSILON {
            panic!("{} != {}", a, b);
        }
    }

    #[test]
    fn project_towards() {
        let dist = 0.001;
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [2.0, 0.0, 0.0];

        let (r1, o1) = p1.project_towards(&p2, 1.0);
        assert_near(&r1, &[2.0, 0.0, 0.0], dist);
        assert_close(o1, 0.0);

        let (r2, o2) = p1.project_towards(&p2, 2.0);
        assert_near(&r2, &[3.0, 0.0, 0.0], dist);
        assert_close(o2, 1.0);

        let (r3, o3) = p1.project_towards(&p2, 0.5);
        assert_near(&r3, &[1.5, 0.0, 0.0], dist);
        assert_close(o3, -0.5);
    }

    fn assert_near<S: Location + Debug>(p1: &S, p2: &S, dist: Precision) {
        if p1.distance_to(p2.location()) >= dist {
            panic!("{:?} not near {:?}", p1, p2);
        }
    }

    fn assert_linestring<S: Location + Debug>(ls1: &[S], ls2: &[S], dist: Precision) {
        assert_eq!(ls1.len(), ls2.len());
        for (p1, p2) in ls1.iter().zip(ls2.iter()) {
            assert_near(p1, p2, dist);
        }
    }

    #[test]
    fn resample_ls() {
        let linestring = make_linestring(&[0., 0., 0.], &[1., 0., 0.], 4);
        let resampled_08 = resample_linestring(&linestring, 0.8);
        assert_linestring(
            &resampled_08,
            &[[0., 0., 0.], [0.8, 0., 0.], [1.6, 0., 0.], [2.4, 0., 0.]],
            0.001,
        );
        let resampled_12 = resample_linestring(&linestring, 1.2);
        assert_linestring(
            &resampled_12,
            &[[0., 0., 0.], [1.2, 0., 0.], [2.4, 0., 0.]],
            0.001,
        );
    }

    #[test]
    fn test_edges_to_tree_constructs() {
        let edges: Vec<(&'static str, Option<&'static str>, ())> = vec![
            ("F", None, ()),
            ("B", Some("F"), ()),
            ("A", Some("B"), ()),
            ("D", Some("B"), ()),
            ("C", Some("D"), ()),
            ("E", Some("D"), ()),
            ("G", Some("F"), ()),
            ("I", Some("G"), ()),
            ("H", Some("I"), ()),
        ];
        let (test_tree, _) = edges_to_tree_with_data(&edges).expect("Couldn't construct");
        print_tree(&test_tree, "TEST");
        let test_dfs: Vec<_> = test_tree
            .root()
            .unwrap()
            .traverse_pre_order()
            .map(|n| n.data().0)
            .collect();

        let (ref_tree, _) = make_topotree();
        let ref_dfs: Vec<_> = ref_tree
            .root()
            .unwrap()
            .traverse_pre_order()
            .map(|n| n.data())
            .collect();

        assert_eq!(format!("{:?}", test_dfs), format!("{:?}", ref_dfs));
    }

    #[test]
    fn test_edges_to_tree_jumbled() {
        let edges: Vec<(&'static str, Option<&'static str>, ())> = vec![
            ("A", Some("B"), ()),
            ("C", Some("D"), ()),
            ("F", None, ()),
            ("G", Some("F"), ()),
            ("E", Some("D"), ()),
            ("D", Some("B"), ()),
            ("I", Some("G"), ()),
            ("H", Some("I"), ()),
            ("B", Some("F"), ()),
        ];
        let (test_tree, _) = edges_to_tree_with_data(&edges).expect("Couldn't construct");
        print_tree(&test_tree, "TEST");
    }
}
