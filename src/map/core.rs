//! This is the core implementation that doesn't depend on the hasher at all.
//!
//! The methods of `RingMapCore` don't use any Hash properties of K.
//!
//! It's cleaner to separate them out, then the compiler checks that we are not
//! using Hash at all in these methods.
//!
//! However, we should probably not let this show in the public API or docs.

mod entry;

pub mod raw_entry_v1;

use hashbrown::hash_table;

use crate::vec_deque::{self, VecDeque};
use crate::TryReserveError;
use core::cmp::Ordering;
use core::ops::RangeBounds;
use core::{iter, mem, slice};

use crate::util::simplify_range;
use crate::{Bucket, Equivalent, HashValue};

type Indices = hash_table::HashTable<OffsetIndex>;
type Entries<K, V> = VecDeque<Bucket<K, V>>;

pub use entry::{Entry, IndexedEntry, OccupiedEntry, VacantEntry};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OffsetIndex {
    raw: usize,
}

impl OffsetIndex {
    #[inline]
    fn new(i: usize, offset: usize) -> Self {
        Self {
            raw: i.wrapping_add(offset),
        }
    }

    #[inline]
    fn get(self, offset: usize) -> usize {
        self.raw.wrapping_sub(offset)
    }
}

/// Core of the map that does not depend on S
#[derive(Debug)]
pub(crate) struct RingMapCore<K, V> {
    /// indices mapping from the entry hash to its index, with an offset.
    /// i.e. `entries[i]` is stored as `i.wrapping_add(offset)` in `indices`.
    indices: Indices,
    /// entries is a dense vec maintaining entry order.
    entries: Entries<K, V>,
    offset: usize,
}

/// Mutable references to the parts of an `RingMapCore`.
///
/// When using `HashTable::find_entry`, that takes hold of `&mut indices`, so we have to borrow our
/// `&mut entries` separately, and there's no way to go back to a `&mut RingMapCore`. So this type
/// is used to implement methods on the split references, and `RingMapCore` can also call those to
/// avoid duplication.
struct RefMut<'a, K, V> {
    indices: &'a mut Indices,
    entries: &'a mut Entries<K, V>,
    offset: &'a mut usize,
}

#[inline(always)]
fn get_hash<K, V>(entries: &Entries<K, V>, offset: usize) -> impl Fn(&OffsetIndex) -> u64 + '_ {
    move |&i| entries[i.get(offset)].hash.get()
}

#[inline]
fn equivalent<'a, K, V, Q: ?Sized + Equivalent<K>>(
    key: &'a Q,
    entries: &'a Entries<K, V>,
    offset: usize,
) -> impl Fn(&OffsetIndex) -> bool + 'a {
    move |&i| Q::equivalent(key, &entries[i.get(offset)].key)
}

#[inline]
fn erase_index(table: &mut Indices, offset: usize, hash: HashValue, index: usize) {
    let needle = OffsetIndex::new(index, offset);
    if let Ok(entry) = table.find_entry(hash.get(), move |&i| i == needle) {
        entry.remove();
    } else if cfg!(debug_assertions) {
        panic!("index not found");
    }
}

#[inline]
fn update_index(table: &mut Indices, offset: usize, hash: HashValue, old: usize, new: usize) {
    let old = OffsetIndex::new(old, offset);
    let index = table
        .find_mut(hash.get(), move |&i| i == old)
        .expect("index not found");
    *index = OffsetIndex::new(new, offset);
}

/// A simple alias to help `clippy::type_complexity`
type Pair<T> = (T, T);

#[inline]
fn len_slices<T>((head, tail): Pair<&[T]>) -> usize {
    head.len() + tail.len()
}

#[inline]
fn iter_slices<T>(
    (head, tail): Pair<&'_ [T]>,
) -> iter::Chain<slice::Iter<'_, T>, slice::Iter<'_, T>> {
    head.iter().chain(tail)
}

#[inline]
fn split_slices<T>((head, tail): Pair<&'_ [T]>, i: usize) -> Pair<Pair<&'_ [T]>> {
    if i <= head.len() {
        let (head, mid) = head.split_at(i);
        ((head, &[]), (mid, tail))
    } else {
        let (mid, tail) = tail.split_at(i - head.len());
        ((head, mid), (tail, &[]))
    }
}

#[inline]
fn split_slices_mut<T>((head, tail): Pair<&'_ mut [T]>, i: usize) -> Pair<Pair<&'_ mut [T]>> {
    if i <= head.len() {
        let (head, mid) = head.split_at_mut(i);
        ((head, &mut []), (mid, tail))
    } else {
        let (mid, tail) = tail.split_at_mut(i - head.len());
        ((head, mid), (tail, &mut []))
    }
}

#[inline]
fn sub_slices_mut<T>(slices: Pair<&'_ mut [T]>, start: usize, end: usize) -> Pair<&'_ mut [T]> {
    let (slices, _) = split_slices_mut(slices, end);
    split_slices_mut(slices, start).1
}

/// Inserts many entries into the indices table without reallocating,
/// and without regard for duplication.
///
/// ***Panics*** if there is not sufficient capacity already.
fn insert_bulk_no_grow<K, V>(indices: &mut Indices, offset: usize, entries: Pair<&[Bucket<K, V>]>) {
    assert!(indices.capacity() - indices.len() >= len_slices(entries));
    for entry in iter_slices(entries) {
        let index = OffsetIndex::new(indices.len(), offset);
        indices.insert_unique(entry.hash.get(), index, |_| unreachable!());
    }
}

impl<K, V> Clone for RingMapCore<K, V>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        let mut new = Self::new();
        new.clone_from(self);
        new
    }

    fn clone_from(&mut self, other: &Self) {
        self.indices.clone_from(&other.indices);
        if self.entries.capacity() < other.entries.len() {
            // If we must resize, match the indices capacity.
            let additional = other.entries.len() - self.entries.len();
            self.borrow_mut().reserve_entries(additional);
        }
        self.entries.clone_from(&other.entries);
        self.offset = other.offset;
    }
}

impl<K, V> crate::Entries for RingMapCore<K, V> {
    type Entry = Bucket<K, V>;

    #[inline]
    fn into_entries(self) -> Entries<K, V> {
        self.entries
    }

    #[inline]
    fn as_entries(&self) -> &Entries<K, V> {
        &self.entries
    }

    #[inline]
    fn as_entries_mut(&mut self) -> &mut Entries<K, V> {
        &mut self.entries
    }

    fn with_contiguous_entries<F>(&mut self, f: F)
    where
        F: FnOnce(&mut [Self::Entry]),
    {
        f(self.entries.make_contiguous());
        self.rebuild_hash_table();
    }
}

impl<K, V> RingMapCore<K, V> {
    /// The maximum capacity before the `entries` allocation would exceed `isize::MAX`.
    const MAX_ENTRIES_CAPACITY: usize = (isize::MAX as usize) / mem::size_of::<Bucket<K, V>>();

    #[inline]
    pub(crate) const fn new() -> Self {
        RingMapCore {
            indices: Indices::new(),
            entries: VecDeque::new(),
            offset: 0,
        }
    }

    #[inline]
    fn borrow_mut(&mut self) -> RefMut<'_, K, V> {
        RefMut::new(&mut self.indices, &mut self.entries, &mut self.offset)
    }

    #[inline]
    pub(crate) fn with_capacity(n: usize) -> Self {
        RingMapCore {
            indices: Indices::with_capacity(n),
            entries: VecDeque::with_capacity(n),
            offset: 0,
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        Ord::min(self.indices.capacity(), self.entries.capacity())
    }

    pub(crate) fn clear(&mut self) {
        self.indices.clear();
        self.entries.clear();
        self.offset = 0;
    }

    pub(crate) fn truncate(&mut self, len: usize) {
        if len < self.len() {
            self.erase_indices(len, self.entries.len());
            self.entries.truncate(len);
        }
    }

    #[track_caller]
    pub(crate) fn drain<R>(&mut self, range: R) -> vec_deque::Drain<'_, Bucket<K, V>>
    where
        R: RangeBounds<usize>,
    {
        let range = simplify_range(range, self.entries.len());
        self.erase_indices(range.start, range.end);
        self.entries.drain(range)
    }

    #[cfg(feature = "rayon")]
    pub(crate) fn par_drain<R>(
        &mut self,
        range: R,
    ) -> rayon::collections::vec_deque::Drain<'_, Bucket<K, V>>
    where
        K: Send,
        V: Send,
        R: RangeBounds<usize>,
    {
        use rayon::iter::ParallelDrainRange;
        let range = simplify_range(range, self.entries.len());
        self.erase_indices(range.start, range.end);
        self.entries.par_drain(range)
    }

    #[track_caller]
    pub(crate) fn split_off(&mut self, at: usize) -> Self {
        let len = self.entries.len();
        assert!(
            at <= len,
            "index out of bounds: the len is {len} but the index is {at}. Expected index <= len"
        );

        self.erase_indices(at, self.entries.len());
        let entries = self.entries.split_off(at);

        let mut indices = Indices::with_capacity(entries.len());
        insert_bulk_no_grow(&mut indices, 0, entries.as_slices());
        Self {
            indices,
            entries,
            offset: 0,
        }
    }

    #[track_caller]
    pub(crate) fn split_splice<R>(&mut self, range: R) -> (Self, vec_deque::IntoIter<Bucket<K, V>>)
    where
        R: RangeBounds<usize>,
    {
        let range = simplify_range(range, self.len());
        self.erase_indices(range.start, self.entries.len());
        let entries = self.entries.split_off(range.end);
        let drained = self.entries.split_off(range.start);

        let mut indices = Indices::with_capacity(entries.len());
        insert_bulk_no_grow(&mut indices, 0, entries.as_slices());
        (
            Self {
                indices,
                entries,
                offset: 0,
            },
            drained.into_iter(),
        )
    }

    /// Append from another map without checking whether items already exist.
    pub(crate) fn append_unchecked(&mut self, other: &mut Self) {
        self.reserve(other.len());
        insert_bulk_no_grow(&mut self.indices, self.offset, other.entries.as_slices());
        self.entries.append(&mut other.entries);
        other.indices.clear();
        other.offset = 0;
    }

    /// Reserve capacity for `additional` more key-value pairs.
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.indices
            .reserve(additional, get_hash(&self.entries, self.offset));
        // Only grow entries if necessary, since we also round up capacity.
        if additional > self.entries.capacity() - self.entries.len() {
            self.borrow_mut().reserve_entries(additional);
        }
    }

    /// Reserve capacity for `additional` more key-value pairs, without over-allocating.
    pub(crate) fn reserve_exact(&mut self, additional: usize) {
        self.indices
            .reserve(additional, get_hash(&self.entries, self.offset));
        self.entries.reserve_exact(additional);
    }

    /// Try to reserve capacity for `additional` more key-value pairs.
    pub(crate) fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.indices
            .try_reserve(additional, get_hash(&self.entries, self.offset))
            .map_err(TryReserveError::from_hashbrown)?;
        // Only grow entries if necessary, since we also round up capacity.
        if additional > self.entries.capacity() - self.entries.len() {
            self.try_reserve_entries(additional)
        } else {
            Ok(())
        }
    }

    /// Try to reserve entries capacity, rounded up to match the indices
    fn try_reserve_entries(&mut self, additional: usize) -> Result<(), TryReserveError> {
        // Use a soft-limit on the maximum capacity, but if the caller explicitly
        // requested more, do it and let them have the resulting error.
        let new_capacity = Ord::min(self.indices.capacity(), Self::MAX_ENTRIES_CAPACITY);
        let try_add = new_capacity - self.entries.len();
        if try_add > additional && self.entries.try_reserve_exact(try_add).is_ok() {
            return Ok(());
        }
        self.entries
            .try_reserve_exact(additional)
            .map_err(TryReserveError::from_alloc)
    }

    /// Try to reserve capacity for `additional` more key-value pairs, without over-allocating.
    pub(crate) fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.indices
            .try_reserve(additional, get_hash(&self.entries, self.offset))
            .map_err(TryReserveError::from_hashbrown)?;
        self.entries
            .try_reserve_exact(additional)
            .map_err(TryReserveError::from_alloc)
    }

    /// Shrink the capacity of the map with a lower bound
    pub(crate) fn shrink_to(&mut self, min_capacity: usize) {
        self.indices
            .shrink_to(min_capacity, get_hash(&self.entries, self.offset));
        self.entries.shrink_to(min_capacity);
    }

    /// Remove the last key-value pair
    pub(crate) fn pop_back(&mut self) -> Option<(K, V)> {
        if let Some(entry) = self.entries.pop_back() {
            let last = self.entries.len();
            erase_index(&mut self.indices, self.offset, entry.hash, last);
            Some((entry.key, entry.value))
        } else {
            None
        }
    }

    /// Remove the first key-value pair
    pub(crate) fn pop_front(&mut self) -> Option<(K, V)> {
        if let Some(entry) = self.entries.pop_front() {
            erase_index(&mut self.indices, self.offset, entry.hash, 0);
            self.offset = self.offset.wrapping_add(1);
            Some((entry.key, entry.value))
        } else {
            None
        }
    }

    /// Return the index in `entries` where an equivalent key can be found
    pub(crate) fn get_index_of<Q>(&self, hash: HashValue, key: &Q) -> Option<usize>
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.entries, self.offset);
        let oi = self.indices.find(hash.get(), eq)?;
        Some(oi.get(self.offset))
    }

    pub(crate) fn push_back(&mut self, hash: HashValue, key: K, value: V) -> (usize, Option<V>)
    where
        K: Eq,
    {
        let eq = equivalent(&key, &self.entries, self.offset);
        let hasher = get_hash(&self.entries, self.offset);
        match self.indices.entry(hash.get(), eq, hasher) {
            hash_table::Entry::Occupied(entry) => {
                let i = entry.get().get(self.offset);
                (i, Some(mem::replace(&mut self.entries[i].value, value)))
            }
            hash_table::Entry::Vacant(entry) => {
                let i = self.entries.len();
                let oi = OffsetIndex::new(i, self.offset);
                entry.insert(oi);
                self.borrow_mut().push_back_entry(hash, key, value);
                debug_assert_eq!(self.indices.len(), self.entries.len());
                (i, None)
            }
        }
    }

    pub(crate) fn push_front(&mut self, hash: HashValue, key: K, value: V) -> (usize, Option<V>)
    where
        K: Eq,
    {
        let eq = equivalent(&key, &self.entries, self.offset);
        let hasher = get_hash(&self.entries, self.offset);
        match self.indices.entry(hash.get(), eq, hasher) {
            hash_table::Entry::Occupied(entry) => {
                let i = entry.get().get(self.offset);
                (i, Some(mem::replace(&mut self.entries[i].value, value)))
            }
            hash_table::Entry::Vacant(entry) => {
                let oi = OffsetIndex::new(usize::MAX, self.offset);
                entry.insert(oi);
                self.borrow_mut().push_front_entry(hash, key, value);
                debug_assert_eq!(self.indices.len(), self.entries.len());
                self.offset = self.offset.wrapping_sub(1); // now MAX is 0
                (0, None)
            }
        }
    }

    /// Same as `insert_full`, except it also replaces the key
    pub(crate) fn replace_full(
        &mut self,
        hash: HashValue,
        key: K,
        value: V,
    ) -> (usize, Option<(K, V)>)
    where
        K: Eq,
    {
        let eq = equivalent(&key, &self.entries, self.offset);
        let hasher = get_hash(&self.entries, self.offset);
        match self.indices.entry(hash.get(), eq, hasher) {
            hash_table::Entry::Occupied(entry) => {
                let i = entry.get().get(self.offset);
                let entry = &mut self.entries[i];
                let kv = (
                    mem::replace(&mut entry.key, key),
                    mem::replace(&mut entry.value, value),
                );
                (i, Some(kv))
            }
            hash_table::Entry::Vacant(entry) => {
                let i = self.entries.len();
                let oi = OffsetIndex::new(i, self.offset);
                entry.insert(oi);
                self.borrow_mut().push_back_entry(hash, key, value);
                debug_assert_eq!(self.indices.len(), self.entries.len());
                (i, None)
            }
        }
    }

    /// Remove an entry by shifting all entries that follow it
    pub(crate) fn shift_remove_full<Q>(&mut self, hash: HashValue, key: &Q) -> Option<(usize, K, V)>
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.entries, self.offset);
        match self.indices.find_entry(hash.get(), eq) {
            Ok(entry) => {
                let (oi, _) = entry.remove();
                let index = oi.get(self.offset);
                let (key, value) = self.borrow_mut().shift_remove_finish(index);
                Some((index, key, value))
            }
            Err(_) => None,
        }
    }

    /// Remove an entry by shifting all entries that follow it
    #[inline]
    pub(crate) fn shift_remove_index(&mut self, index: usize) -> Option<(K, V)> {
        self.borrow_mut().shift_remove_index(index)
    }

    #[inline]
    #[track_caller]
    pub(super) fn move_index(&mut self, from: usize, to: usize) {
        self.borrow_mut().move_index(from, to);
    }

    #[inline]
    #[track_caller]
    pub(crate) fn swap_indices(&mut self, a: usize, b: usize) {
        self.borrow_mut().swap_indices(a, b);
    }

    /// Remove an entry by swapping it with the last
    pub(crate) fn swap_remove_back_full<Q>(
        &mut self,
        hash: HashValue,
        key: &Q,
    ) -> Option<(usize, K, V)>
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.entries, self.offset);
        match self.indices.find_entry(hash.get(), eq) {
            Ok(entry) => {
                let (oi, _) = entry.remove();
                let index = oi.get(self.offset);
                let (key, value) = self.borrow_mut().swap_remove_back_finish(index);
                Some((index, key, value))
            }
            Err(_) => None,
        }
    }

    /// Remove an entry by swapping it with the last
    #[inline]
    pub(crate) fn swap_remove_back_index(&mut self, index: usize) -> Option<(K, V)> {
        self.borrow_mut().swap_remove_back_index(index)
    }

    /// Remove an entry by swapping it with the first
    pub(crate) fn swap_remove_front_full<Q>(
        &mut self,
        hash: HashValue,
        key: &Q,
    ) -> Option<(usize, K, V)>
    where
        Q: ?Sized + Equivalent<K>,
    {
        let eq = equivalent(key, &self.entries, self.offset);
        match self.indices.find_entry(hash.get(), eq) {
            Ok(entry) => {
                let (oi, _) = entry.remove();
                let index = oi.get(self.offset);
                let (key, value) = self.borrow_mut().swap_remove_front_finish(index);
                Some((index, key, value))
            }
            Err(_) => None,
        }
    }

    /// Remove an entry by swapping it with the first
    #[inline]
    pub(crate) fn swap_remove_front_index(&mut self, index: usize) -> Option<(K, V)> {
        self.borrow_mut().swap_remove_front_index(index)
    }

    /// Erase `start..end` from `indices`, and shift `end..` indices down to `start..`
    ///
    /// All of these items should still be at their original location in `entries`.
    /// This is used by `drain`, which will let `VecDeque::drain` do the work on `entries`.
    fn erase_indices(&mut self, start: usize, end: usize) {
        let (init, end_entries) = split_slices(self.entries.as_slices(), end);
        let (start_entries, erased_entries) = split_slices(init, start);

        let erased = len_slices(erased_entries);
        let end_len = len_slices(end_entries);
        let half_capacity = self.indices.capacity() / 2;

        // Use a heuristic between different strategies
        if erased == 0 {
            // Degenerate case, nothing to do
        } else if start + end_len < half_capacity && start < erased {
            // Reinsert everything, as there are few kept indices
            self.indices.clear();
            self.offset = 0;

            // Reinsert start indices, then end indices
            insert_bulk_no_grow(&mut self.indices, 0, start_entries);
            insert_bulk_no_grow(&mut self.indices, 0, end_entries);
        } else if start + erased < half_capacity || erased + end_len < half_capacity {
            // Find each affected index, as there are few to adjust

            // Find erased indices
            for (i, entry) in (start..).zip(iter_slices(erased_entries)) {
                erase_index(&mut self.indices, self.offset, entry.hash, i);
            }

            if start < end_len {
                // Find start indices and shift them up
                let start_indices = (erased..end).zip(0..start).rev();
                for ((new, old), entry) in start_indices.zip(iter_slices(start_entries).rev()) {
                    update_index(&mut self.indices, self.offset, entry.hash, old, new);
                }
                self.offset = self.offset.wrapping_add(erased);
            } else {
                // Find end indices and shift them down
                for ((new, old), entry) in (start..).zip(end..).zip(iter_slices(end_entries)) {
                    update_index(&mut self.indices, self.offset, entry.hash, old, new);
                }
            }
        } else {
            // Sweep the whole table for adjustments
            let offset = self.offset;
            self.indices.retain(move |i| {
                let index = i.get(offset);
                if index >= end {
                    *i = OffsetIndex::new(index - erased, offset);
                    true
                } else {
                    index < start
                }
            });
        }

        debug_assert_eq!(self.indices.len(), start + end_len);
    }

    pub(crate) fn retain_in_order<F>(&mut self, mut keep: F)
    where
        F: FnMut(&mut K, &mut V) -> bool,
    {
        self.entries
            .retain_mut(|entry| keep(&mut entry.key, &mut entry.value));
        if self.entries.len() < self.indices.len() {
            self.rebuild_hash_table();
        }
    }

    fn rebuild_hash_table(&mut self) {
        self.indices.clear();
        self.offset = 0;
        insert_bulk_no_grow(&mut self.indices, 0, self.entries.as_slices());
    }

    pub(crate) fn reverse(&mut self) {
        let mut iter = self.entries.iter_mut();
        while let (Some(head), Some(tail)) = (iter.next(), iter.next_back()) {
            mem::swap(head, tail);
        }

        // No need to save hash indices, can easily calculate what they should
        // be, given that this is an in-place reversal.
        let len = self.entries.len();
        for i in &mut self.indices {
            *i = OffsetIndex::new(len - i.get(self.offset) - 1, 0);
        }
        self.offset = 0;
    }

    pub(super) fn binary_search_keys(&self, x: &K) -> Result<usize, usize>
    where
        K: Ord,
    {
        self.binary_search_by(|p, _| p.cmp(x))
    }

    pub(super) fn binary_search_by<'a, F>(&'a self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&'a K, &'a V) -> Ordering,
    {
        self.entries.binary_search_by(move |a| f(&a.key, &a.value))
    }

    pub(super) fn binary_search_by_key<'a, B, F>(&'a self, b: &B, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&'a K, &'a V) -> B,
        B: Ord,
    {
        self.binary_search_by(|k, v| f(k, v).cmp(b))
    }

    #[must_use]
    pub(super) fn partition_point<P>(&self, mut pred: P) -> usize
    where
        P: FnMut(&K, &V) -> bool,
    {
        self.entries
            .partition_point(move |a| pred(&a.key, &a.value))
    }
}

/// Reserve entries capacity, rounded up to match the indices (via `try_capacity`).
fn reserve_entries<K, V>(entries: &mut Entries<K, V>, additional: usize, try_capacity: usize) {
    // Use a soft-limit on the maximum capacity, but if the caller explicitly
    // requested more, do it and let them have the resulting panic.
    let try_capacity = try_capacity.min(RingMapCore::<K, V>::MAX_ENTRIES_CAPACITY);
    let try_add = try_capacity - entries.len();
    if try_add > additional && entries.try_reserve_exact(try_add).is_ok() {
        return;
    }
    entries.reserve_exact(additional);
}

impl<'a, K, V> RefMut<'a, K, V> {
    #[inline]
    fn new(
        indices: &'a mut Indices,
        entries: &'a mut Entries<K, V>,
        offset: &'a mut usize,
    ) -> Self {
        Self {
            indices,
            entries,
            offset,
        }
    }

    /// Reserve entries capacity, rounded up to match the indices
    #[inline]
    fn reserve_entries(&mut self, additional: usize) {
        reserve_entries(self.entries, additional, self.indices.capacity());
    }

    /// Append a key-value pair to `entries`,
    /// *without* checking whether it already exists.
    fn push_front_entry(&mut self, hash: HashValue, key: K, value: V) {
        if self.entries.len() == self.entries.capacity() {
            // Reserve our own capacity synced to the indices,
            // rather than letting `VecDeque::push` just double it.
            self.reserve_entries(1);
        }
        self.entries.push_front(Bucket { hash, key, value });
    }

    /// Append a key-value pair to `entries`,
    /// *without* checking whether it already exists.
    fn push_back_entry(&mut self, hash: HashValue, key: K, value: V) {
        if self.entries.len() == self.entries.capacity() {
            // Reserve our own capacity synced to the indices,
            // rather than letting `VecDeque::push` just double it.
            self.reserve_entries(1);
        }
        self.entries.push_back(Bucket { hash, key, value });
    }

    fn push_front_unique(self, hash: HashValue, key: K, value: V) -> OccupiedEntry<'a, K, V> {
        let oi = OffsetIndex::new(usize::MAX, *self.offset);
        let entry =
            self.indices
                .insert_unique(hash.get(), oi, get_hash(self.entries, *self.offset));
        if self.entries.len() == self.entries.capacity() {
            // We can't call `indices.capacity()` while this `entry` has borrowed it, so we'll have
            // to amortize growth on our own. It's still an improvement over the basic `Vec::push`
            // doubling though, since we also consider `MAX_ENTRIES_CAPACITY`.
            reserve_entries(self.entries, 1, 2 * self.entries.capacity());
        }
        self.entries.push_front(Bucket { hash, key, value });
        *self.offset = self.offset.wrapping_sub(1); // now MAX is 0
        OccupiedEntry::new(self.entries, self.offset, entry)
    }

    fn push_back_unique(self, hash: HashValue, key: K, value: V) -> OccupiedEntry<'a, K, V> {
        let i = self.indices.len();
        debug_assert_eq!(i, self.entries.len());
        let oi = OffsetIndex::new(i, *self.offset);
        let entry =
            self.indices
                .insert_unique(hash.get(), oi, get_hash(self.entries, *self.offset));
        if self.entries.len() == self.entries.capacity() {
            // We can't call `indices.capacity()` while this `entry` has borrowed it, so we'll have
            // to amortize growth on our own. It's still an improvement over the basic `Vec::push`
            // doubling though, since we also consider `MAX_ENTRIES_CAPACITY`.
            reserve_entries(self.entries, 1, 2 * self.entries.capacity());
        }
        self.entries.push_back(Bucket { hash, key, value });
        OccupiedEntry::new(self.entries, self.offset, entry)
    }

    fn shift_insert_unique(&mut self, index: usize, hash: HashValue, key: K, value: V) {
        let end = self.indices.len();
        assert!(index <= end);
        // Increment others first so we don't have duplicate indices.
        self.increment_indices(index, end);
        let entries = &*self.entries;
        let offset = *self.offset;
        let oi = OffsetIndex::new(index, offset);
        self.indices.insert_unique(hash.get(), oi, move |&i| {
            // Adjust for the incremented indices to find hashes.
            let i = i.get(offset);
            debug_assert_ne!(i, index);
            let i = if i < index { i } else { i - 1 };
            entries[i].hash.get()
        });
        if self.entries.len() == self.entries.capacity() {
            // Reserve our own capacity synced to the indices,
            // rather than letting `Vec::insert` just double it.
            self.reserve_entries(1);
        }
        self.entries.insert(index, Bucket { hash, key, value });
    }

    /// Remove an entry by shifting all entries that follow it
    fn shift_remove_index(&mut self, index: usize) -> Option<(K, V)> {
        match self.entries.get(index) {
            Some(entry) => {
                erase_index(self.indices, *self.offset, entry.hash, index);
                Some(self.shift_remove_finish(index))
            }
            None => None,
        }
    }

    /// Remove an entry by shifting all entries that follow it
    ///
    /// The index should already be removed from `self.indices`.
    fn shift_remove_finish(&mut self, index: usize) -> (K, V) {
        // Correct indices that point to the entries that followed the removed entry.
        self.decrement_indices(index + 1, self.entries.len());

        // Use VecDeque::remove to actually remove the entry.
        let entry = self.entries.remove(index).unwrap();
        (entry.key, entry.value)
    }

    /// Remove an entry by swapping it with the last
    fn swap_remove_back_index(&mut self, index: usize) -> Option<(K, V)> {
        match self.entries.get(index) {
            Some(entry) => {
                erase_index(self.indices, *self.offset, entry.hash, index);
                Some(self.swap_remove_back_finish(index))
            }
            None => None,
        }
    }

    /// Finish removing an entry by swapping it with the last
    ///
    /// The index should already be removed from `self.indices`.
    fn swap_remove_back_finish(&mut self, index: usize) -> (K, V) {
        // use swap_remove_back, but then we need to update the index that points
        // to the other entry that has to move
        let entry = self.entries.swap_remove_back(index).unwrap();

        // correct index that points to the entry that had to swap places
        if let Some(entry) = self.entries.get(index) {
            // was not last element
            // examine new element in `index` and find it in indices
            let last = self.entries.len();
            update_index(self.indices, *self.offset, entry.hash, last, index);
        }

        (entry.key, entry.value)
    }

    /// Remove an entry by swapping it with the first
    fn swap_remove_front_index(&mut self, index: usize) -> Option<(K, V)> {
        match self.entries.get(index) {
            Some(entry) => {
                erase_index(self.indices, *self.offset, entry.hash, index);
                Some(self.swap_remove_front_finish(index))
            }
            None => None,
        }
    }

    /// Finish removing an entry by swapping it with the first
    ///
    /// The index should already be removed from `self.indices`.
    fn swap_remove_front_finish(&mut self, index: usize) -> (K, V) {
        // use swap_remove_front, but then we need to update the index that points
        // to the other entry that has to move
        let entry = self.entries.swap_remove_front(index).unwrap();

        // correct index that points to the entry that had to swap places
        if index > 0 {
            // was not first element
            if let Some(entry) = self.entries.get(index - 1) {
                // examine new element in `index` and find it in indices
                update_index(self.indices, *self.offset, entry.hash, 0, index);
            }
        }
        *self.offset = self.offset.wrapping_add(1);

        (entry.key, entry.value)
    }

    /// Decrement all indices in the range `start..end`.
    ///
    /// The index `start - 1` should not exist in `self.indices`.
    /// All entries should still be in their original positions.
    fn decrement_indices(&mut self, start: usize, end: usize) {
        let (init, end_entries) = split_slices(self.entries.as_slices(), end);
        let (mut start_entries, target_entries) = split_slices(init, start);
        (start_entries, _) = split_slices(start_entries, start - 1);

        let start_len = len_slices(start_entries);
        let target_len = len_slices(target_entries);
        let end_len = len_slices(end_entries);
        let half_capacity = self.indices.capacity() / 2;

        // Use a heuristic between a full sweep vs. a `find()` for every shifted item.
        if target_len <= start_len + end_len {
            // Find each entry in range to decrement its index.
            for (i, entry) in (start..end).zip(iter_slices(target_entries)) {
                update_index(self.indices, *self.offset, entry.hash, i, i - 1);
            }
        } else if start_len + end_len < half_capacity {
            // Find each entry outside the range and increment them instead.
            // (in reverse to avoid ever having duplicates)
            for (i, entry) in (end..end + end_len)
                .rev()
                .zip(iter_slices(end_entries).rev())
            {
                update_index(self.indices, *self.offset, entry.hash, i, i + 1);
            }
            for (i, entry) in (0..start_len).rev().zip(iter_slices(start_entries).rev()) {
                update_index(self.indices, *self.offset, entry.hash, i, i + 1);
            }
            *self.offset = self.offset.wrapping_add(1);
        } else {
            // Shift all indices in range.
            for i in &mut *self.indices {
                let index = i.get(*self.offset);
                if start <= index && index < end {
                    *i = OffsetIndex::new(index - 1, *self.offset);
                }
            }
        }
    }

    /// Increment all indices in the range `start..end`.
    ///
    /// The index `end` should not exist in `self.indices`.
    /// All entries should still be in their original positions.
    fn increment_indices(&mut self, start: usize, end: usize) {
        let (init, mut end_entries) = split_slices(self.entries.as_slices(), end);
        let (start_entries, target_entries) = split_slices(init, start);
        if end < self.entries.len() {
            (_, end_entries) = split_slices(end_entries, 1);
        }

        let start_len = len_slices(start_entries);
        let target_len = len_slices(target_entries);
        let end_len = len_slices(end_entries);
        let half_capacity = self.indices.capacity() / 2;

        // Use a heuristic between a full sweep vs. a `find()` for every shifted item.
        if target_len <= start_len + end_len {
            // Find each entry in range to increment its index, updated in reverse so
            // we never have duplicated indices that might have a hash collision.
            for (i, entry) in (start..end).rev().zip(iter_slices(target_entries).rev()) {
                update_index(self.indices, *self.offset, entry.hash, i, i + 1);
            }
        } else if start_len + end_len < half_capacity {
            // Find each entry outside the range and decrement them instead.
            *self.offset = self.offset.wrapping_sub(1);
            for (i, entry) in (0..).zip(iter_slices(start_entries)) {
                update_index(self.indices, *self.offset, entry.hash, i + 1, i);
            }
            for (i, entry) in (end + 1..).zip(iter_slices(end_entries)) {
                update_index(self.indices, *self.offset, entry.hash, i + 1, i);
            }
        } else {
            // Shift all indices in range.
            for i in &mut *self.indices {
                let index = i.get(*self.offset);
                if start <= index && index < end {
                    *i = OffsetIndex::new(index + 1, *self.offset);
                }
            }
        }
    }

    #[track_caller]
    fn move_index(&mut self, from: usize, to: usize) {
        let from_hash = self.entries[from].hash;
        let _ = self.entries[to]; // explicit bounds check
        if from != to {
            // Use a sentinel index so other indices don't collide.
            let orig_offset = *self.offset;
            let sentinel = isize::MIN as usize;
            update_index(self.indices, orig_offset, from_hash, from, sentinel);

            // Update all other indices and rotate the entry positions.
            if from < to {
                self.decrement_indices(from + 1, to + 1);
                // self.entries[from..=to].rotate_left(1);
                if from == 0 {
                    let entry = self.entries.pop_front().unwrap();
                    self.entries.insert(to, entry);
                } else if to + 1 == self.entries.len() {
                    let entry = self.entries.remove(from).unwrap();
                    self.entries.push_back(entry);
                } else {
                    match sub_slices_mut(self.entries.as_mut_slices(), from, to + 1) {
                        (xs, []) | ([], xs) => xs.rotate_left(1),
                        (xs, ys) => {
                            mem::swap(&mut xs[0], &mut ys[0]);
                            xs.rotate_left(1);
                            ys.rotate_left(1);
                        }
                    }
                }
            } else if to < from {
                self.increment_indices(to, from);
                // self.entries[to..=from].rotate_right(1);
                if to == 0 {
                    let entry = self.entries.remove(from).unwrap();
                    self.entries.push_front(entry);
                } else if from + 1 == self.entries.len() {
                    let entry = self.entries.pop_back().unwrap();
                    self.entries.insert(to, entry);
                } else {
                    match sub_slices_mut(self.entries.as_mut_slices(), to, from + 1) {
                        (xs, []) | ([], xs) => xs.rotate_right(1),
                        (xs, ys) => {
                            mem::swap(&mut xs[xs.len() - 1], &mut ys[ys.len() - 1]);
                            xs.rotate_right(1);
                            ys.rotate_right(1);
                        }
                    }
                }
            }

            // Change the sentinel index to its final position.
            // (taking care if `decrement`/`increment_indices` changed the offset)
            let sentinel = sentinel
                .wrapping_add(orig_offset)
                .wrapping_sub(*self.offset);
            update_index(self.indices, *self.offset, from_hash, sentinel, to);
        }
    }

    #[track_caller]
    fn swap_indices(&mut self, a: usize, b: usize) {
        // If they're equal and in-bounds, there's nothing to do.
        if a == b && a < self.entries.len() {
            return;
        }

        // We'll get a "nice" bounds-check from indexing `entries`,
        // and then we expect to find it in the table as well.
        let oa = OffsetIndex::new(a, *self.offset);
        let ob = OffsetIndex::new(b, *self.offset);
        match self.indices.get_many_mut(
            [self.entries[a].hash.get(), self.entries[b].hash.get()],
            move |i, &x| if i == 0 { x == oa } else { x == ob },
        ) {
            [Some(ref_a), Some(ref_b)] => {
                mem::swap(ref_a, ref_b);
                self.entries.swap(a, b);
            }
            _ => panic!("indices not found"),
        }
    }

    fn binary_search_keys(&self, x: &K) -> Result<usize, usize>
    where
        K: Ord,
    {
        self.binary_search_by(|p, _| p.cmp(x))
    }

    fn binary_search_by<'b, F>(&'b self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&'b K, &'b V) -> Ordering,
    {
        self.entries.binary_search_by(move |a| f(&a.key, &a.value))
    }
}

#[test]
fn assert_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RingMapCore<i32, i32>>();
    assert_send_sync::<Entry<'_, i32, i32>>();
    assert_send_sync::<IndexedEntry<'_, i32, i32>>();
    assert_send_sync::<raw_entry_v1::RawEntryMut<'_, i32, i32, ()>>();
}
