//! Parallel iterator types for [`RingMap`] with [`rayon`][::rayon].
//!
//! You will rarely need to interact with this module directly unless you need to name one of the
//! iterator types.

use super::collect;
use rayon::iter::plumbing::{Consumer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{BuildHasher, Hash};
use core::ops::RangeBounds;

use crate::map::Slice;
use crate::Bucket;
use crate::Entries;
use crate::RingMap;

impl<K, V, S> IntoParallelIterator for RingMap<K, V, S>
where
    K: Send,
    V: Send,
{
    type Item = (K, V);
    type Iter = IntoParIter<K, V>;

    fn into_par_iter(self) -> Self::Iter {
        IntoParIter {
            entries: self.into_entries(),
        }
    }
}

impl<K, V> IntoParallelIterator for Box<Slice<K, V>>
where
    K: Send,
    V: Send,
{
    type Item = (K, V);
    type Iter = IntoParIter<K, V>;

    fn into_par_iter(self) -> Self::Iter {
        IntoParIter {
            entries: self.into_entries(),
        }
    }
}

/// A parallel owning iterator over the entries of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::into_par_iter`] method
/// (provided by rayon's [`IntoParallelIterator`] trait). See its documentation for more.
pub struct IntoParIter<K, V> {
    entries: VecDeque<Bucket<K, V>>,
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for IntoParIter<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.entries.iter().map(Bucket::refs);
        f.debug_list().entries(iter).finish()
    }
}

impl<K: Send, V: Send> ParallelIterator for IntoParIter<K, V> {
    type Item = (K, V);

    parallel_iterator_methods!(Bucket::key_value);
}

impl<K: Send, V: Send> IndexedParallelIterator for IntoParIter<K, V> {
    indexed_parallel_iterator_methods!(Bucket::key_value);
}

/// Internal iterator over `VecDeque` slices
pub(super) struct ParBuckets<'a, K, V> {
    head: &'a [Bucket<K, V>],
    tail: &'a [Bucket<K, V>],
}

impl<'a, K, V> ParBuckets<'a, K, V> {
    pub(super) fn new(entries: &'a VecDeque<Bucket<K, V>>) -> Self {
        Self::from_slices(entries.as_slices())
    }

    pub(super) fn from_slices((head, tail): (&'a [Bucket<K, V>], &'a [Bucket<K, V>])) -> Self {
        Self { head, tail }
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &Bucket<K, V>> {
        self.head.iter().chain(self.tail)
    }
}

impl<K, V> Clone for ParBuckets<'_, K, V> {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

impl<'a, K: Sync, V: Sync> ParallelIterator for ParBuckets<'a, K, V> {
    type Item = &'a Bucket<K, V>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        self.head
            .par_iter()
            .chain(self.tail)
            .drive_unindexed(consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<'a, K: Sync, V: Sync> IndexedParallelIterator for ParBuckets<'a, K, V> {
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        self.head.par_iter().chain(self.tail).drive(consumer)
    }

    fn len(&self) -> usize {
        self.head.len() + self.tail.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        self.head
            .par_iter()
            .chain(self.tail)
            .with_producer(callback)
    }
}

impl<'a, K, V, S> IntoParallelIterator for &'a RingMap<K, V, S>
where
    K: Sync,
    V: Sync,
{
    type Item = (&'a K, &'a V);
    type Iter = ParIter<'a, K, V>;

    fn into_par_iter(self) -> Self::Iter {
        ParIter {
            entries: ParBuckets::new(self.as_entries()),
        }
    }
}

impl<'a, K, V> IntoParallelIterator for &'a Slice<K, V>
where
    K: Sync,
    V: Sync,
{
    type Item = (&'a K, &'a V);
    type Iter = ParIter<'a, K, V>;

    fn into_par_iter(self) -> Self::Iter {
        ParIter {
            entries: ParBuckets::from_slices((&self.entries, &[])),
        }
    }
}

/// A parallel iterator over the entries of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::par_iter`] method
/// (provided by rayon's [`IntoParallelRefIterator`] trait). See its documentation for more.
///
/// [`RingMap::par_iter`]: ../struct.RingMap.html#method.par_iter
pub struct ParIter<'a, K, V> {
    entries: ParBuckets<'a, K, V>,
}

impl<K, V> Clone for ParIter<'_, K, V> {
    fn clone(&self) -> Self {
        ParIter {
            entries: self.entries.clone(),
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for ParIter<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.entries.iter().map(Bucket::refs);
        f.debug_list().entries(iter).finish()
    }
}

impl<'a, K: Sync, V: Sync> ParallelIterator for ParIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    parallel_iterator_methods!(Bucket::refs);
}

impl<K: Sync, V: Sync> IndexedParallelIterator for ParIter<'_, K, V> {
    indexed_parallel_iterator_methods!(Bucket::refs);
}

/// Internal iterator over `VecDeque` mutable slices
struct ParBucketsMut<'a, K, V> {
    head: &'a mut [Bucket<K, V>],
    tail: &'a mut [Bucket<K, V>],
}

impl<'a, K, V> ParBucketsMut<'a, K, V> {
    fn new(entries: &'a mut VecDeque<Bucket<K, V>>) -> Self {
        Self::from_mut_slices(entries.as_mut_slices())
    }

    fn from_mut_slices((head, tail): (&'a mut [Bucket<K, V>], &'a mut [Bucket<K, V>])) -> Self {
        Self { head, tail }
    }

    fn iter(&self) -> impl Iterator<Item = &Bucket<K, V>> {
        self.head.iter().chain(&*self.tail)
    }
}

impl<'a, K: Send, V: Send> ParallelIterator for ParBucketsMut<'a, K, V> {
    type Item = &'a mut Bucket<K, V>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        self.head
            .par_iter_mut()
            .chain(self.tail)
            .drive_unindexed(consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<'a, K: Send, V: Send> IndexedParallelIterator for ParBucketsMut<'a, K, V> {
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        self.head.par_iter_mut().chain(self.tail).drive(consumer)
    }

    fn len(&self) -> usize {
        self.head.len() + self.tail.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        self.head
            .par_iter_mut()
            .chain(self.tail)
            .with_producer(callback)
    }
}

impl<'a, K, V, S> IntoParallelIterator for &'a mut RingMap<K, V, S>
where
    K: Sync + Send,
    V: Send,
{
    type Item = (&'a K, &'a mut V);
    type Iter = ParIterMut<'a, K, V>;

    fn into_par_iter(self) -> Self::Iter {
        ParIterMut {
            entries: ParBucketsMut::new(self.as_entries_mut()),
        }
    }
}

impl<'a, K, V> IntoParallelIterator for &'a mut Slice<K, V>
where
    K: Sync + Send,
    V: Send,
{
    type Item = (&'a K, &'a mut V);
    type Iter = ParIterMut<'a, K, V>;

    fn into_par_iter(self) -> Self::Iter {
        ParIterMut {
            entries: ParBucketsMut::from_mut_slices((&mut self.entries, &mut [])),
        }
    }
}

/// A parallel mutable iterator over the entries of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::par_iter_mut`] method
/// (provided by rayon's [`IntoParallelRefMutIterator`] trait). See its documentation for more.
///
/// [`RingMap::par_iter_mut`]: ../struct.RingMap.html#method.par_iter_mut
pub struct ParIterMut<'a, K, V> {
    entries: ParBucketsMut<'a, K, V>,
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for ParIterMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.entries.iter().map(Bucket::refs);
        f.debug_list().entries(iter).finish()
    }
}

impl<'a, K: Sync + Send, V: Send> ParallelIterator for ParIterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    parallel_iterator_methods!(Bucket::ref_mut);
}

impl<K: Sync + Send, V: Send> IndexedParallelIterator for ParIterMut<'_, K, V> {
    indexed_parallel_iterator_methods!(Bucket::ref_mut);
}

impl<'a, K, V, S> ParallelDrainRange<usize> for &'a mut RingMap<K, V, S>
where
    K: Send,
    V: Send,
{
    type Item = (K, V);
    type Iter = ParDrain<'a, K, V>;

    fn par_drain<R: RangeBounds<usize>>(self, range: R) -> Self::Iter {
        ParDrain {
            entries: self.core.par_drain(range),
        }
    }
}

/// A parallel draining iterator over the entries of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::par_drain`] method
/// (provided by rayon's [`ParallelDrainRange`] trait). See its documentation for more.
///
/// [`RingMap::par_drain`]: ../struct.RingMap.html#method.par_drain
pub struct ParDrain<'a, K: Send, V: Send> {
    entries: rayon::collections::vec_deque::Drain<'a, Bucket<K, V>>,
}

impl<K: Send, V: Send> ParallelIterator for ParDrain<'_, K, V> {
    type Item = (K, V);

    parallel_iterator_methods!(Bucket::key_value);
}

impl<K: Send, V: Send> IndexedParallelIterator for ParDrain<'_, K, V> {
    indexed_parallel_iterator_methods!(Bucket::key_value);
}

/// Parallel iterator methods and other parallel methods.
///
/// The following methods **require crate feature `"rayon"`**.
///
/// See also the `IntoParallelIterator` implementations.
impl<K, V, S> RingMap<K, V, S>
where
    K: Sync,
    V: Sync,
{
    /// Return a parallel iterator over the keys of the map.
    ///
    /// While parallel iterators can process items in any order, their relative order
    /// in the map is still preserved for operations like `reduce` and `collect`.
    pub fn par_keys(&self) -> ParKeys<'_, K, V> {
        ParKeys {
            entries: ParBuckets::new(self.as_entries()),
        }
    }

    /// Return a parallel iterator over the values of the map.
    ///
    /// While parallel iterators can process items in any order, their relative order
    /// in the map is still preserved for operations like `reduce` and `collect`.
    pub fn par_values(&self) -> ParValues<'_, K, V> {
        ParValues {
            entries: ParBuckets::new(self.as_entries()),
        }
    }
}

/// Parallel iterator methods and other parallel methods.
///
/// The following methods **require crate feature `"rayon"`**.
///
/// See also the `IntoParallelIterator` implementations.
impl<K, V> Slice<K, V>
where
    K: Sync,
    V: Sync,
{
    /// Return a parallel iterator over the keys of the map slice.
    ///
    /// While parallel iterators can process items in any order, their relative order
    /// in the slice is still preserved for operations like `reduce` and `collect`.
    pub fn par_keys(&self) -> ParKeys<'_, K, V> {
        ParKeys {
            entries: ParBuckets::from_slices((&self.entries, &[])),
        }
    }

    /// Return a parallel iterator over the values of the map slice.
    ///
    /// While parallel iterators can process items in any order, their relative order
    /// in the slice is still preserved for operations like `reduce` and `collect`.
    pub fn par_values(&self) -> ParValues<'_, K, V> {
        ParValues {
            entries: ParBuckets::from_slices((&self.entries, &[])),
        }
    }
}

impl<K, V, S> RingMap<K, V, S>
where
    K: PartialEq + Sync,
    V: Sync,
{
    /// Returns `true` if `self` contains all of the same key-value pairs as `other`,
    /// in the same indexed order, determined in parallel.
    pub fn par_eq<S2>(&self, other: &RingMap<K, V, S2>) -> bool
    where
        V: PartialEq,
    {
        self.len() == other.len() && self.par_iter().eq(other)
    }
}

/// A parallel iterator over the keys of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::par_keys`] method.
/// See its documentation for more.
pub struct ParKeys<'a, K, V> {
    entries: ParBuckets<'a, K, V>,
}

impl<K, V> Clone for ParKeys<'_, K, V> {
    fn clone(&self) -> Self {
        ParKeys {
            entries: self.entries.clone(),
        }
    }
}

impl<K: fmt::Debug, V> fmt::Debug for ParKeys<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.entries.iter().map(Bucket::key_ref);
        f.debug_list().entries(iter).finish()
    }
}

impl<'a, K: Sync, V: Sync> ParallelIterator for ParKeys<'a, K, V> {
    type Item = &'a K;

    parallel_iterator_methods!(Bucket::key_ref);
}

impl<K: Sync, V: Sync> IndexedParallelIterator for ParKeys<'_, K, V> {
    indexed_parallel_iterator_methods!(Bucket::key_ref);
}

/// A parallel iterator over the values of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::par_values`] method.
/// See its documentation for more.
pub struct ParValues<'a, K, V> {
    entries: ParBuckets<'a, K, V>,
}

impl<K, V> Clone for ParValues<'_, K, V> {
    fn clone(&self) -> Self {
        ParValues {
            entries: self.entries.clone(),
        }
    }
}

impl<K, V: fmt::Debug> fmt::Debug for ParValues<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.entries.iter().map(Bucket::value_ref);
        f.debug_list().entries(iter).finish()
    }
}

impl<'a, K: Sync, V: Sync> ParallelIterator for ParValues<'a, K, V> {
    type Item = &'a V;

    parallel_iterator_methods!(Bucket::value_ref);
}

impl<K: Sync, V: Sync> IndexedParallelIterator for ParValues<'_, K, V> {
    indexed_parallel_iterator_methods!(Bucket::value_ref);
}

impl<K, V, S> RingMap<K, V, S>
where
    K: Send,
    V: Send,
{
    /// Return a parallel iterator over mutable references to the values of the map
    ///
    /// While parallel iterators can process items in any order, their relative order
    /// in the map is still preserved for operations like `reduce` and `collect`.
    pub fn par_values_mut(&mut self) -> ParValuesMut<'_, K, V> {
        ParValuesMut {
            entries: ParBucketsMut::new(self.as_entries_mut()),
        }
    }
}

impl<K, V> Slice<K, V>
where
    K: Send,
    V: Send,
{
    /// Return a parallel iterator over mutable references to the the values of the map slice.
    ///
    /// While parallel iterators can process items in any order, their relative order
    /// in the slice is still preserved for operations like `reduce` and `collect`.
    pub fn par_values_mut(&mut self) -> ParValuesMut<'_, K, V> {
        ParValuesMut {
            entries: ParBucketsMut::from_mut_slices((&mut self.entries, &mut [])),
        }
    }
}

impl<K, V, S> RingMap<K, V, S>
where
    K: Send,
    V: Send,
{
    /// Sort the map’s key-value pairs in parallel, by the default ordering of the keys.
    pub fn par_sort_keys(&mut self)
    where
        K: Ord,
    {
        self.with_contiguous_entries(|entries| {
            entries.par_sort_by(|a, b| K::cmp(&a.key, &b.key));
        });
    }

    /// Sort the map’s key-value pairs in place and in parallel, using the comparison
    /// function `cmp`.
    ///
    /// The comparison function receives two key and value pairs to compare (you
    /// can sort by keys or values or their combination as needed).
    pub fn par_sort_by<F>(&mut self, cmp: F)
    where
        F: Fn(&K, &V, &K, &V) -> Ordering + Sync,
    {
        self.with_contiguous_entries(|entries| {
            entries.par_sort_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        });
    }

    /// Sort the key-value pairs of the map in parallel and return a by-value parallel
    /// iterator of the key-value pairs with the result.
    pub fn par_sorted_by<F>(self, cmp: F) -> IntoParIter<K, V>
    where
        F: Fn(&K, &V, &K, &V) -> Ordering + Sync,
    {
        let mut entries = self.into_entries();
        entries
            .make_contiguous()
            .par_sort_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        IntoParIter { entries }
    }

    /// Sort the map's key-value pairs in parallel, by the default ordering of the keys.
    pub fn par_sort_unstable_keys(&mut self)
    where
        K: Ord,
    {
        self.with_contiguous_entries(|entries| {
            entries.par_sort_unstable_by(|a, b| K::cmp(&a.key, &b.key));
        });
    }

    /// Sort the map's key-value pairs in place and in parallel, using the comparison
    /// function `cmp`.
    ///
    /// The comparison function receives two key and value pairs to compare (you
    /// can sort by keys or values or their combination as needed).
    pub fn par_sort_unstable_by<F>(&mut self, cmp: F)
    where
        F: Fn(&K, &V, &K, &V) -> Ordering + Sync,
    {
        self.with_contiguous_entries(|entries| {
            entries.par_sort_unstable_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        });
    }

    /// Sort the key-value pairs of the map in parallel and return a by-value parallel
    /// iterator of the key-value pairs with the result.
    pub fn par_sorted_unstable_by<F>(self, cmp: F) -> IntoParIter<K, V>
    where
        F: Fn(&K, &V, &K, &V) -> Ordering + Sync,
    {
        let mut entries = self.into_entries();
        entries
            .make_contiguous()
            .par_sort_unstable_by(move |a, b| cmp(&a.key, &a.value, &b.key, &b.value));
        IntoParIter { entries }
    }

    /// Sort the map’s key-value pairs in place and in parallel, using a sort-key extraction
    /// function.
    pub fn par_sort_by_cached_key<T, F>(&mut self, sort_key: F)
    where
        T: Ord + Send,
        F: Fn(&K, &V) -> T + Sync,
    {
        self.with_contiguous_entries(move |entries| {
            entries.par_sort_by_cached_key(move |a| sort_key(&a.key, &a.value));
        });
    }
}

/// A parallel mutable iterator over the values of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::par_values_mut`] method.
/// See its documentation for more.
pub struct ParValuesMut<'a, K, V> {
    entries: ParBucketsMut<'a, K, V>,
}

impl<K, V: fmt::Debug> fmt::Debug for ParValuesMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.entries.iter().map(Bucket::value_ref);
        f.debug_list().entries(iter).finish()
    }
}

impl<'a, K: Send, V: Send> ParallelIterator for ParValuesMut<'a, K, V> {
    type Item = &'a mut V;

    parallel_iterator_methods!(Bucket::value_mut);
}

impl<K: Send, V: Send> IndexedParallelIterator for ParValuesMut<'_, K, V> {
    indexed_parallel_iterator_methods!(Bucket::value_mut);
}

impl<K, V, S> FromParallelIterator<(K, V)> for RingMap<K, V, S>
where
    K: Eq + Hash + Send,
    V: Send,
    S: BuildHasher + Default + Send,
{
    fn from_par_iter<I>(iter: I) -> Self
    where
        I: IntoParallelIterator<Item = (K, V)>,
    {
        let list = collect(iter);
        let len = list.iter().map(Vec::len).sum();
        let mut map = Self::with_capacity_and_hasher(len, S::default());
        for vec in list {
            map.extend(vec);
        }
        map
    }
}

impl<K, V, S> ParallelExtend<(K, V)> for RingMap<K, V, S>
where
    K: Eq + Hash + Send,
    V: Send,
    S: BuildHasher + Send,
{
    fn par_extend<I>(&mut self, iter: I)
    where
        I: IntoParallelIterator<Item = (K, V)>,
    {
        for vec in collect(iter) {
            self.extend(vec);
        }
    }
}

impl<'a, K: 'a, V: 'a, S> ParallelExtend<(&'a K, &'a V)> for RingMap<K, V, S>
where
    K: Copy + Eq + Hash + Send + Sync,
    V: Copy + Send + Sync,
    S: BuildHasher + Send,
{
    fn par_extend<I>(&mut self, iter: I)
    where
        I: IntoParallelIterator<Item = (&'a K, &'a V)>,
    {
        for vec in collect(iter) {
            self.extend(vec);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_order() {
        let insert = [0, 4, 2, 12, 8, 7, 11, 5, 3, 17, 19, 22, 23];
        let mut map = RingMap::new();

        for &elt in &insert {
            map.insert(elt, ());
        }

        assert_eq!(map.par_keys().count(), map.len());
        assert_eq!(map.par_keys().count(), insert.len());
        insert.par_iter().zip(map.par_keys()).for_each(|(a, b)| {
            assert_eq!(a, b);
        });
        (0..insert.len())
            .into_par_iter()
            .zip(map.par_keys())
            .for_each(|(i, k)| {
                assert_eq!(map.get_index(i).unwrap().0, k);
            });
    }

    #[test]
    fn partial_eq_and_eq() {
        let mut map_a = RingMap::new();
        map_a.insert(1, "1");
        map_a.insert(2, "2");
        let mut map_b = map_a.clone();
        assert!(map_a.par_eq(&map_b));
        map_b.swap_remove_back(&1);
        assert!(!map_a.par_eq(&map_b));
        map_b.insert(3, "3");
        assert!(!map_a.par_eq(&map_b));
    }

    #[test]
    fn extend() {
        let mut map = RingMap::new();
        map.par_extend(vec![(&1, &2), (&3, &4)]);
        map.par_extend(vec![(5, 6)]);
        assert_eq!(
            map.into_par_iter().collect::<Vec<_>>(),
            vec![(1, 2), (3, 4), (5, 6)]
        );
    }

    #[test]
    fn keys() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: RingMap<_, _> = vec.into_par_iter().collect();
        let keys: Vec<_> = map.par_keys().copied().collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn values() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: RingMap<_, _> = vec.into_par_iter().collect();
        let values: Vec<_> = map.par_values().copied().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn values_mut() {
        let vec = vec![(1, 1), (2, 2), (3, 3)];
        let mut map: RingMap<_, _> = vec.into_par_iter().collect();
        map.par_values_mut().for_each(|value| *value *= 2);
        let values: Vec<_> = map.par_values().copied().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&2));
        assert!(values.contains(&4));
        assert!(values.contains(&6));
    }
}
