/// Disjoint-set / union-find with path compression and union by size.
pub struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    /// Find the root of the set containing `x`, with path compression.
    pub fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression
        let mut current = x;
        while self.parent[current] != root {
            let next = self.parent[current];
            self.parent[current] = root;
            current = next;
        }
        root
    }

    /// Union the sets containing `x` and `y`. Returns `true` if they were disjoint.
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }
        // Union by size: attach smaller tree under larger
        if self.size[rx] < self.size[ry] {
            self.parent[rx] = ry;
            self.size[ry] += self.size[rx];
        } else {
            self.parent[ry] = rx;
            self.size[rx] += self.size[ry];
        }
        true
    }

    pub fn size_of(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.size[root]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut uf = UnionFind::new(5);
        assert!(uf.union(0, 1));
        assert!(uf.union(2, 3));
        assert_eq!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(0), uf.find(2));
        assert!(uf.union(1, 3));
        assert_eq!(uf.find(0), uf.find(3));
        assert!(!uf.union(0, 2)); // already connected
        assert_eq!(uf.size_of(0), 4);
        assert_eq!(uf.size_of(4), 1);
    }
}
