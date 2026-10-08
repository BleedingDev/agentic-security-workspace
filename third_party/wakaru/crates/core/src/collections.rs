//! Hash tables for the engine's internal analysis.
//!
//! Keys are almost always `(Atom, SyntaxContext)` binding identities, atoms,
//! or spans: small, fixed-size values whose `Hash` impls already write a
//! precomputed word. `FxHasher` turns that word into a slot in a couple of
//! multiplies, where the standard library's SipHash spends most of the lookup
//! on keyed rounds that only matter for untrusted input. Iteration order was
//! never something callers could rely on (the default hasher is randomly
//! seeded per process), so the swap changes cost, not results.
//!
//! Public engine APIs that callers outside this crate populate keep the
//! standard-library types; everything internal uses these aliases.

pub type HashMap<K, V> = std::collections::HashMap<K, V, rustc_hash::FxBuildHasher>;
pub type HashSet<T> = std::collections::HashSet<T, rustc_hash::FxBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use swc_core::atoms::Atom;
    use swc_core::common::SyntaxContext;

    #[test]
    fn aliases_use_the_fx_hasher_and_keep_std_semantics() {
        assert!(std::any::type_name::<HashMap<u8, u8>>().contains("FxBuildHasher"));
        assert!(std::any::type_name::<HashSet<u8>>().contains("FxBuildHasher"));

        let a = (Atom::from("a"), SyntaxContext::empty());
        let b = (Atom::from("b"), SyntaxContext::empty());
        let mut map: HashMap<(Atom, SyntaxContext), usize> = HashMap::default();
        *map.entry(a.clone()).or_default() += 1;
        *map.entry(a.clone()).or_default() += 1;
        map.insert(b.clone(), 7);
        assert_eq!(map.get(&a), Some(&2));
        assert_eq!(map.get(&b), Some(&7));

        let set: HashSet<(Atom, SyntaxContext)> = HashSet::from_iter([a.clone(), a.clone(), b]);
        assert_eq!(set.len(), 2);
        let sized: HashSet<u32> = HashSet::with_capacity_and_hasher(8, Default::default());
        assert!(sized.capacity() >= 8);
        let collected: HashSet<(Atom, SyntaxContext)> = map.keys().cloned().collect();
        assert_eq!(collected, set);
    }
}
