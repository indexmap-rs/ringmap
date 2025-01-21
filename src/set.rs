//! A hash set implemented using [`RingMap`]

mod iter;
mod mutable;
mod slice;

#[cfg(test)]
mod tests;

pub use self::iter::{
    Difference, Drain, Intersection, IntoIter, Iter, Splice, SymmetricDifference, Union,
};
pub use self::mutable::MutableValues;
pub use self::slice::Slice;

#[cfg(feature = "rayon")]
pub use crate::rayon::set as rayon;
use crate::TryReserveError;

#[cfg(feature = "std")]
use std::collections::hash_map::RandomState;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{BuildHasher, Hash, Hasher};
use core::ops::{BitAnd, BitOr, BitXor, Index, RangeBounds, Sub};

use super::{Entries, Equivalent, RingMap};

type Bucket<T> = super::Bucket<T, ()>;

/// A hash set where the iteration order of the values is independent of their
/// hash values.
///
/// The interface is closely compatible with the standard
/// [`HashSet`][std::collections::HashSet],
/// but also has additional features.
///
/// # Order
///
/// The values have a consistent order that is determined by the sequence of
/// insertion and removal calls on the set. The order does not depend on the
/// values or the hash function at all. Note that insertion order and value
/// are not affected if a re-insertion is attempted once an element is
/// already present.
///
/// All iterators traverse the set *in order*.  Set operation iterators like
/// [`RingSet::union`] produce a concatenated order, as do their matching "bitwise"
/// operators.  See their documentation for specifics.
///
/// The insertion order is preserved, with **notable exceptions** like the
/// [`.swap_remove_front()`][Self::swap_remove_front] or [`.swap_remove_back()`][Self::swap_remove_back] methods.
/// Methods such as [`.sort_by()`][Self::sort_by] of
/// course result in a new order, depending on the sorting order.
///
/// # Indices
///
/// The values are indexed in a compact range without holes in the range
/// `0..self.len()`. For example, the method `.get_full` looks up the index for
/// a value, and the method `.get_index` looks up the value by index.
///
/// # Complexity
///
/// Internally, `RingSet<T, S>` just holds an [`RingMap<T, (), S>`](RingMap). Thus the complexity
/// of the two are the same for most methods.
///
/// # Examples
///
/// ```
/// use ringmap::RingSet;
///
/// // Collects which letters appear in a sentence.
/// let letters: RingSet<_> = "a short treatise on fungi".chars().collect();
///
/// assert!(letters.contains(&'s'));
/// assert!(letters.contains(&'t'));
/// assert!(letters.contains(&'u'));
/// assert!(!letters.contains(&'y'));
/// ```
#[cfg(feature = "std")]
pub struct RingSet<T, S = RandomState> {
    pub(crate) map: RingMap<T, (), S>,
}
#[cfg(not(feature = "std"))]
pub struct RingSet<T, S> {
    pub(crate) map: RingMap<T, (), S>,
}

impl<T, S> Clone for RingSet<T, S>
where
    T: Clone,
    S: Clone,
{
    fn clone(&self) -> Self {
        RingSet {
            map: self.map.clone(),
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.map.clone_from(&other.map);
    }
}

impl<T, S> Entries for RingSet<T, S> {
    type Entry = Bucket<T>;

    #[inline]
    fn into_entries(self) -> VecDeque<Self::Entry> {
        self.map.into_entries()
    }

    #[inline]
    fn as_entries(&self) -> &VecDeque<Self::Entry> {
        self.map.as_entries()
    }

    #[inline]
    fn as_entries_mut(&mut self) -> &mut VecDeque<Self::Entry> {
        self.map.as_entries_mut()
    }

    fn with_contiguous_entries<F>(&mut self, f: F)
    where
        F: FnOnce(&mut [Self::Entry]),
    {
        self.map.with_contiguous_entries(f);
    }
}

impl<T, S> fmt::Debug for RingSet<T, S>
where
    T: fmt::Debug,
{
    #[cfg(not(feature = "test_debug"))]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }

    #[cfg(feature = "test_debug")]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Let the inner `RingSet` print all of its details
        f.debug_struct("RingSet").field("map", &self.map).finish()
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<T> RingSet<T> {
    /// Create a new set. (Does not allocate.)
    pub fn new() -> Self {
        RingSet {
            map: RingMap::new(),
        }
    }

    /// Create a new set with capacity for `n` elements.
    /// (Does not allocate if `n` is zero.)
    ///
    /// Computes in **O(n)** time.
    pub fn with_capacity(n: usize) -> Self {
        RingSet {
            map: RingMap::with_capacity(n),
        }
    }
}

impl<T, S> RingSet<T, S> {
    /// Create a new set with capacity for `n` elements.
    /// (Does not allocate if `n` is zero.)
    ///
    /// Computes in **O(n)** time.
    pub fn with_capacity_and_hasher(n: usize, hash_builder: S) -> Self {
        RingSet {
            map: RingMap::with_capacity_and_hasher(n, hash_builder),
        }
    }

    /// Create a new set with `hash_builder`.
    ///
    /// This function is `const`, so it
    /// can be called in `static` contexts.
    pub const fn with_hasher(hash_builder: S) -> Self {
        RingSet {
            map: RingMap::with_hasher(hash_builder),
        }
    }

    /// Return the number of elements the set can hold without reallocating.
    ///
    /// This number is a lower bound; the set might be able to hold more,
    /// but is guaranteed to be able to hold at least this many.
    ///
    /// Computes in **O(1)** time.
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }

    /// Return a reference to the set's `BuildHasher`.
    pub fn hasher(&self) -> &S {
        self.map.hasher()
    }

    /// Return the number of elements in the set.
    ///
    /// Computes in **O(1)** time.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if the set contains no elements.
    ///
    /// Computes in **O(1)** time.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Return an iterator over the values of the set, in their order
    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self.as_entries())
    }

    /// Remove all elements in the set, while preserving its capacity.
    ///
    /// Computes in **O(n)** time.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Shortens the set, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater than the set's current length, this has no effect.
    pub fn truncate(&mut self, len: usize) {
        self.map.truncate(len);
    }

    /// Clears the `RingSet` in the given index range, returning those values
    /// as a drain iterator.
    ///
    /// The range may be any type that implements [`RangeBounds<usize>`],
    /// including all of the `std::ops::Range*` types, or even a tuple pair of
    /// `Bound` start and end values. To drain the set entirely, use `RangeFull`
    /// like `set.drain(..)`.
    ///
    /// This shifts down all entries following the drained range to fill the
    /// gap, and keeps the allocated memory for reuse.
    ///
    /// ***Panics*** if the starting point is greater than the end point or if
    /// the end point is greater than the length of the set.
    #[track_caller]
    pub fn drain<R>(&mut self, range: R) -> Drain<'_, T>
    where
        R: RangeBounds<usize>,
    {
        Drain::new(self.map.core.drain(range))
    }

    /// Splits the collection into two at the given index.
    ///
    /// Returns a newly allocated set containing the elements in the range
    /// `[at, len)`. After the call, the original set will be left containing
    /// the elements `[0, at)` with its previous capacity unchanged.
    ///
    /// ***Panics*** if `at > len`.
    #[track_caller]
    pub fn split_off(&mut self, at: usize) -> Self
    where
        S: Clone,
    {
        Self {
            map: self.map.split_off(at),
        }
    }

    /// Reserve capacity for `additional` more values.
    ///
    /// Computes in **O(n)** time.
    pub fn reserve(&mut self, additional: usize) {
        self.map.reserve(additional);
    }

    /// Reserve capacity for `additional` more values, without over-allocating.
    ///
    /// Unlike `reserve`, this does not deliberately over-allocate the entry capacity to avoid
    /// frequent re-allocations. However, the underlying data structures may still have internal
    /// capacity requirements, and the allocator itself may give more space than requested, so this
    /// cannot be relied upon to be precisely minimal.
    ///
    /// Computes in **O(n)** time.
    pub fn reserve_exact(&mut self, additional: usize) {
        self.map.reserve_exact(additional);
    }

    /// Try to reserve capacity for `additional` more values.
    ///
    /// Computes in **O(n)** time.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.map.try_reserve(additional)
    }

    /// Try to reserve capacity for `additional` more values, without over-allocating.
    ///
    /// Unlike `try_reserve`, this does not deliberately over-allocate the entry capacity to avoid
    /// frequent re-allocations. However, the underlying data structures may still have internal
    /// capacity requirements, and the allocator itself may give more space than requested, so this
    /// cannot be relied upon to be precisely minimal.
    ///
    /// Computes in **O(n)** time.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.map.try_reserve_exact(additional)
    }

    /// Shrink the capacity of the set as much as possible.
    ///
    /// Computes in **O(n)** time.
    pub fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit();
    }

    /// Shrink the capacity of the set with a lower limit.
    ///
    /// Computes in **O(n)** time.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.map.shrink_to(min_capacity);
    }
}

impl<T, S> RingSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher,
{
    /// Insert the value into the set.
    ///
    /// If an equivalent item already exists in the set, it returns
    /// `false` leaving the original value in the set and without
    /// altering its insertion order. Otherwise, it inserts the new
    /// item and returns `true`.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn insert(&mut self, value: T) -> bool {
        self.map.insert(value, ()).is_none()
    }

    /// Insert the value into the set, and get its index.
    ///
    /// If an equivalent item already exists in the set, it returns
    /// the index of the existing item and `false`, leaving the
    /// original value in the set and without altering its insertion
    /// order. Otherwise, it inserts the new item and returns the index
    /// of the inserted item and `true`.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn insert_full(&mut self, value: T) -> (usize, bool) {
        let (index, existing) = self.map.insert_full(value, ());
        (index, existing.is_none())
    }

    /// Appends the value into the set, and get its index.
    ///
    /// If an equivalent item already exists in the set, it returns
    /// the index of the existing item and `false`, leaving the
    /// original value in the set and without altering its insertion
    /// order. Otherwise, it inserts the new item at the back and
    /// returns the index of the inserted item and `true`.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn push_back(&mut self, value: T) -> (usize, bool) {
        let (index, existing) = self.map.push_back(value, ());
        (index, existing.is_none())
    }

    /// Prepends the value into the set, and get its index.
    ///
    /// If an equivalent item already exists in the set, it returns
    /// the index of the existing item and `false`, leaving the
    /// original value in the set and without altering its insertion
    /// order. Otherwise, it inserts the new item at the front and
    /// returns the index of the inserted item and `true`.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn push_front(&mut self, value: T) -> (usize, bool) {
        let (index, existing) = self.map.push_front(value, ());
        (index, existing.is_none())
    }

    /// Insert the value into the set at its ordered position among sorted values.
    ///
    /// This is equivalent to finding the position with
    /// [`binary_search`][Self::binary_search], and if needed calling
    /// [`insert_before`][Self::insert_before] for a new value.
    ///
    /// If the sorted item is found in the set, it returns the index of that
    /// existing item and `false`, without any change. Otherwise, it inserts the
    /// new item and returns its sorted index and `true`.
    ///
    /// If the existing items are **not** already sorted, then the insertion
    /// index is unspecified (like [`slice::binary_search`]), but the value
    /// is moved to or inserted at that position regardless.
    ///
    /// Computes in **O(n)** time (average). Instead of repeating calls to
    /// `insert_sorted`, it may be faster to call batched [`insert`][Self::insert]
    /// or [`extend`][Self::extend] and only call [`sort`][Self::sort] or
    /// [`sort_unstable`][Self::sort_unstable] once.
    pub fn insert_sorted(&mut self, value: T) -> (usize, bool)
    where
        T: Ord,
    {
        let (index, existing) = self.map.insert_sorted(value, ());
        (index, existing.is_none())
    }

    /// Insert the value into the set before the value at the given index, or at the end.
    ///
    /// If an equivalent item already exists in the set, it returns `false` leaving the
    /// original value in the set, but moved to the new position. The returned index
    /// will either be the given index or one less, depending on how the value moved.
    /// (See [`shift_insert`](Self::shift_insert) for different behavior here.)
    ///
    /// Otherwise, it inserts the new value exactly at the given index and returns `true`.
    ///
    /// ***Panics*** if `index` is out of bounds.
    /// Valid indices are `0..=set.len()` (inclusive).
    ///
    /// Computes in **O(n)** time (average).
    ///
    /// # Examples
    ///
    /// ```
    /// use ringmap::RingSet;
    /// let mut set: RingSet<char> = ('a'..='z').collect();
    ///
    /// // The new value '*' goes exactly at the given index.
    /// assert_eq!(set.get_index_of(&'*'), None);
    /// assert_eq!(set.insert_before(10, '*'), (10, true));
    /// assert_eq!(set.get_index_of(&'*'), Some(10));
    ///
    /// // Moving the value 'a' up will shift others down, so this moves *before* 10 to index 9.
    /// assert_eq!(set.insert_before(10, 'a'), (9, false));
    /// assert_eq!(set.get_index_of(&'a'), Some(9));
    /// assert_eq!(set.get_index_of(&'*'), Some(10));
    ///
    /// // Moving the value 'z' down will shift others up, so this moves to exactly 10.
    /// assert_eq!(set.insert_before(10, 'z'), (10, false));
    /// assert_eq!(set.get_index_of(&'z'), Some(10));
    /// assert_eq!(set.get_index_of(&'*'), Some(11));
    ///
    /// // Moving or inserting before the endpoint is also valid.
    /// assert_eq!(set.len(), 27);
    /// assert_eq!(set.insert_before(set.len(), '*'), (26, false));
    /// assert_eq!(set.get_index_of(&'*'), Some(26));
    /// assert_eq!(set.insert_before(set.len(), '+'), (27, true));
    /// assert_eq!(set.get_index_of(&'+'), Some(27));
    /// assert_eq!(set.len(), 28);
    /// ```
    #[track_caller]
    pub fn insert_before(&mut self, index: usize, value: T) -> (usize, bool) {
        let (index, existing) = self.map.insert_before(index, value, ());
        (index, existing.is_none())
    }

    /// Insert the value into the set at the given index.
    ///
    /// If an equivalent item already exists in the set, it returns `false` leaving
    /// the original value in the set, but moved to the given index.
    /// Note that existing values **cannot** be moved to `index == set.len()`!
    /// (See [`insert_before`](Self::insert_before) for different behavior here.)
    ///
    /// Otherwise, it inserts the new value at the given index and returns `true`.
    ///
    /// ***Panics*** if `index` is out of bounds.
    /// Valid indices are `0..set.len()` (exclusive) when moving an existing value, or
    /// `0..=set.len()` (inclusive) when inserting a new value.
    ///
    /// Computes in **O(n)** time (average).
    ///
    /// # Examples
    ///
    /// ```
    /// use ringmap::RingSet;
    /// let mut set: RingSet<char> = ('a'..='z').collect();
    ///
    /// // The new value '*' goes exactly at the given index.
    /// assert_eq!(set.get_index_of(&'*'), None);
    /// assert_eq!(set.shift_insert(10, '*'), true);
    /// assert_eq!(set.get_index_of(&'*'), Some(10));
    ///
    /// // Moving the value 'a' up to 10 will shift others down, including the '*' that was at 10.
    /// assert_eq!(set.shift_insert(10, 'a'), false);
    /// assert_eq!(set.get_index_of(&'a'), Some(10));
    /// assert_eq!(set.get_index_of(&'*'), Some(9));
    ///
    /// // Moving the value 'z' down to 9 will shift others up, including the '*' that was at 9.
    /// assert_eq!(set.shift_insert(9, 'z'), false);
    /// assert_eq!(set.get_index_of(&'z'), Some(9));
    /// assert_eq!(set.get_index_of(&'*'), Some(10));
    ///
    /// // Existing values can move to len-1 at most, but new values can insert at the endpoint.
    /// assert_eq!(set.len(), 27);
    /// assert_eq!(set.shift_insert(set.len() - 1, '*'), false);
    /// assert_eq!(set.get_index_of(&'*'), Some(26));
    /// assert_eq!(set.shift_insert(set.len(), '+'), true);
    /// assert_eq!(set.get_index_of(&'+'), Some(27));
    /// assert_eq!(set.len(), 28);
    /// ```
    ///
    /// ```should_panic
    /// use ringmap::RingSet;
    /// let mut set: RingSet<char> = ('a'..='z').collect();
    ///
    /// // This is an invalid index for moving an existing value!
    /// set.shift_insert(set.len(), 'a');
    /// ```
    #[track_caller]
    pub fn shift_insert(&mut self, index: usize, value: T) -> bool {
        self.map.shift_insert(index, value, ()).is_none()
    }

    /// Adds a value to the set, replacing the existing value, if any, that is
    /// equal to the given one, without altering its insertion order. Returns
    /// the replaced value.
    ///
    /// Computes in **O(1)** time (average).
    pub fn replace(&mut self, value: T) -> Option<T> {
        self.replace_full(value).1
    }

    /// Adds a value to the set, replacing the existing value, if any, that is
    /// equal to the given one, without altering its insertion order. Returns
    /// the index of the item and its replaced value.
    ///
    /// Computes in **O(1)** time (average).
    pub fn replace_full(&mut self, value: T) -> (usize, Option<T>) {
        let hash = self.map.hash(&value);
        match self.map.core.replace_full(hash, value, ()) {
            (i, Some((replaced, ()))) => (i, Some(replaced)),
            (i, None) => (i, None),
        }
    }

    /// Return an iterator over the values that are in `self` but not `other`.
    ///
    /// Values are produced in the same order that they appear in `self`.
    pub fn difference<'a, S2>(&'a self, other: &'a RingSet<T, S2>) -> Difference<'a, T, S2>
    where
        S2: BuildHasher,
    {
        Difference::new(self, other)
    }

    /// Return an iterator over the values that are in `self` or `other`,
    /// but not in both.
    ///
    /// Values from `self` are produced in their original order, followed by
    /// values from `other` in their original order.
    pub fn symmetric_difference<'a, S2>(
        &'a self,
        other: &'a RingSet<T, S2>,
    ) -> SymmetricDifference<'a, T, S, S2>
    where
        S2: BuildHasher,
    {
        SymmetricDifference::new(self, other)
    }

    /// Return an iterator over the values that are in both `self` and `other`.
    ///
    /// Values are produced in the same order that they appear in `self`.
    pub fn intersection<'a, S2>(&'a self, other: &'a RingSet<T, S2>) -> Intersection<'a, T, S2>
    where
        S2: BuildHasher,
    {
        Intersection::new(self, other)
    }

    /// Return an iterator over all values that are in `self` or `other`.
    ///
    /// Values from `self` are produced in their original order, followed by
    /// values that are unique to `other` in their original order.
    pub fn union<'a, S2>(&'a self, other: &'a RingSet<T, S2>) -> Union<'a, T, S>
    where
        S2: BuildHasher,
    {
        Union::new(self, other)
    }

    /// Creates a splicing iterator that replaces the specified range in the set
    /// with the given `replace_with` iterator and yields the removed items.
    /// `replace_with` does not need to be the same length as `range`.
    ///
    /// The `range` is removed even if the iterator is not consumed until the
    /// end. It is unspecified how many elements are removed from the set if the
    /// `Splice` value is leaked.
    ///
    /// The input iterator `replace_with` is only consumed when the `Splice`
    /// value is dropped. If a value from the iterator matches an existing entry
    /// in the set (outside of `range`), then the original will be unchanged.
    /// Otherwise, the new value will be inserted in the replaced `range`.
    ///
    /// ***Panics*** if the starting point is greater than the end point or if
    /// the end point is greater than the length of the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use ringmap::RingSet;
    ///
    /// let mut set = RingSet::from([0, 1, 2, 3, 4]);
    /// let new = [5, 4, 3, 2, 1];
    /// let removed: Vec<_> = set.splice(2..4, new).collect();
    ///
    /// // 1 and 4 kept their positions, while 5, 3, and 2 were newly inserted.
    /// assert!(set.into_iter().eq([0, 1, 5, 3, 2, 4]));
    /// assert_eq!(removed, &[2, 3]);
    /// ```
    #[track_caller]
    pub fn splice<R, I>(&mut self, range: R, replace_with: I) -> Splice<'_, I::IntoIter, T, S>
    where
        R: RangeBounds<usize>,
        I: IntoIterator<Item = T>,
    {
        Splice::new(self, range, replace_with.into_iter())
    }

    /// Moves all values from `other` into `self`, leaving `other` empty.
    ///
    /// This is equivalent to calling [`insert`][Self::insert] for each value
    /// from `other` in order, which means that values that already exist
    /// in `self` are unchanged in their current position.
    ///
    /// See also [`union`][Self::union] to iterate the combined values by
    /// reference, without modifying `self` or `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ringmap::RingSet;
    ///
    /// let mut a = RingSet::from([3, 2, 1]);
    /// let mut b = RingSet::from([3, 4, 5]);
    /// let old_capacity = b.capacity();
    ///
    /// a.append(&mut b);
    ///
    /// assert_eq!(a.len(), 5);
    /// assert_eq!(b.len(), 0);
    /// assert_eq!(b.capacity(), old_capacity);
    ///
    /// assert!(a.iter().eq(&[3, 2, 1, 4, 5]));
    /// ```
    pub fn append<S2>(&mut self, other: &mut RingSet<T, S2>) {
        self.map.append(&mut other.map);
    }
}

impl<T, S> RingSet<T, S>
where
    S: BuildHasher,
{
    /// Return `true` if an equivalent to `value` exists in the set.
    ///
    /// Computes in **O(1)** time (average).
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.contains_key(value)
    }

    /// Return a reference to the value stored in the set, if it is present,
    /// else `None`.
    ///
    /// Computes in **O(1)** time (average).
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.get_key_value(value).map(|(x, &())| x)
    }

    /// Return item index and value
    pub fn get_full<Q>(&self, value: &Q) -> Option<(usize, &T)>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.get_full(value).map(|(i, x, &())| (i, x))
    }

    /// Return item index, if it exists in the set
    ///
    /// Computes in **O(1)** time (average).
    pub fn get_index_of<Q>(&self, value: &Q) -> Option<usize>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.get_index_of(value)
    }

    /// Remove the value from the set, and return `true` if it was present.
    ///
    /// Like [`VecDeque::remove`], the value is removed by shifting all of the
    /// elements either before or after it, preserving their relative order.
    /// **This perturbs the index of all of the following elements!**
    ///
    /// Return `false` if `value` was not in the set.
    ///
    /// Computes in **O(n)** time (average).
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.remove(value).is_some()
    }

    /// Removes and returns the value in the set, if any, that is equal to the
    /// given one.
    ///
    /// Like [`VecDeque::remove`], the value is removed by shifting all of the
    /// elements either before or after it, preserving their relative order.
    /// **This perturbs the index of all of the following elements!**
    ///
    /// Return `None` if `value` was not in the set.
    ///
    /// Computes in **O(n)** time (average).
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.remove_entry(value).map(|(x, ())| x)
    }

    /// Remove the value from the set return it and the index it had.
    ///
    /// Like [`VecDeque::remove`], the value is removed by shifting all of the
    /// elements either before or after it, preserving their relative order.
    /// **This perturbs the index of all of the following elements!**
    ///
    /// Return `None` if `value` was not in the set.
    pub fn remove_full<Q>(&mut self, value: &Q) -> Option<(usize, T)>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.remove_full(value).map(|(i, x, ())| (i, x))
    }

    /// Remove the value from the set, and return `true` if it was present.
    ///
    /// Like [`VecDeque::swap_remove_back`], the value is removed by swapping it with the
    /// last element of the set and popping it off. **This perturbs
    /// the position of what used to be the last element!**
    ///
    /// Return `false` if `value` was not in the set.
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_remove_back<Q>(&mut self, value: &Q) -> bool
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.swap_remove_back(value).is_some()
    }

    /// Removes and returns the value in the set, if any, that is equal to the
    /// given one.
    ///
    /// Like [`VecDeque::swap_remove_back`], the value is removed by swapping it with the
    /// last element of the set and popping it off. **This perturbs
    /// the position of what used to be the last element!**
    ///
    /// Return `None` if `value` was not in the set.
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_take_back<Q>(&mut self, value: &Q) -> Option<T>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.swap_remove_back_entry(value).map(|(x, ())| x)
    }

    /// Remove the value from the set return it and the index it had.
    ///
    /// Like [`VecDeque::swap_remove_back`], the value is removed by swapping it with the
    /// last element of the set and popping it off. **This perturbs
    /// the position of what used to be the last element!**
    ///
    /// Return `None` if `value` was not in the set.
    pub fn swap_remove_back_full<Q>(&mut self, value: &Q) -> Option<(usize, T)>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map
            .swap_remove_back_full(value)
            .map(|(i, x, ())| (i, x))
    }

    /// Remove the value from the set, and return `true` if it was present.
    ///
    /// Like [`VecDeque::swap_remove_front`], the value is removed by swapping it with the
    /// first element of the set and popping it off. **This perturbs
    /// the position of what used to be the first element!**
    ///
    /// Return `false` if `value` was not in the set.
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_remove_front<Q>(&mut self, value: &Q) -> bool
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.swap_remove_front(value).is_some()
    }

    /// Removes and returns the value in the set, if any, that is equal to the
    /// given one.
    ///
    /// Like [`VecDeque::swap_remove_front`], the value is removed by swapping it with the
    /// first element of the set and popping it off. **This perturbs
    /// the position of what used to be the first element!**
    ///
    /// Return `None` if `value` was not in the set.
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_take_front<Q>(&mut self, value: &Q) -> Option<T>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map.swap_remove_front_entry(value).map(|(x, ())| x)
    }

    /// Remove the value from the set return it and the index it had.
    ///
    /// Like [`VecDeque::swap_remove_front`], the value is removed by swapping it with the
    /// first element of the set and popping it off. **This perturbs
    /// the position of what used to be the first element!**
    ///
    /// Return `None` if `value` was not in the set.
    pub fn swap_remove_front_full<Q>(&mut self, value: &Q) -> Option<(usize, T)>
    where
        Q: ?Sized + Hash + Equivalent<T>,
    {
        self.map
            .swap_remove_front_full(value)
            .map(|(i, x, ())| (i, x))
    }
}

impl<T, S> RingSet<T, S> {
    /// Remove the last value
    ///
    /// This preserves the order of the remaining elements.
    ///
    /// Computes in **O(1)** time (average).
    #[doc(alias = "pop", alias = "pop_last")] // like `Vec` and `BTreeSet`
    pub fn pop_back(&mut self) -> Option<T> {
        self.map.pop_back().map(|(x, ())| x)
    }

    /// Remove the first value
    ///
    /// This preserves the order of the remaining elements.
    ///
    /// Computes in **O(1)** time (average).
    #[doc(alias = "pop_first")] // like `BTreeSet`
    pub fn pop_front(&mut self) -> Option<T> {
        self.map.pop_front().map(|(x, ())| x)
    }

    /// Scan through each value in the set and keep those where the
    /// closure `keep` returns `true`.
    ///
    /// The elements are visited in order, and remaining elements keep their
    /// order.
    ///
    /// Computes in **O(n)** time (average).
    pub fn retain<F>(&mut self, mut keep: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.map.retain(move |x, &mut ()| keep(x))
    }

    /// Sort the set’s values by their default ordering.
    ///
    /// This is a stable sort -- but equivalent values should not normally coexist in
    /// a set at all, so [`sort_unstable`][Self::sort_unstable] is preferred
    /// because it is generally faster and doesn't allocate auxiliary memory.
    ///
    /// See [`sort_by`](Self::sort_by) for details.
    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.map.sort_keys()
    }

    /// Sort the set’s values in place using the comparison function `cmp`.
    ///
    /// Computes in **O(n log n)** time and **O(n)** space. The sort is stable.
    pub fn sort_by<F>(&mut self, mut cmp: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.map.sort_by(move |a, _, b, _| cmp(a, b));
    }

    /// Sort the values of the set and return a by-value iterator of
    /// the values with the result.
    ///
    /// The sort is stable.
    pub fn sorted_by<F>(self, mut cmp: F) -> IntoIter<T>
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        let mut entries = self.into_entries();
        entries
            .make_contiguous()
            .sort_by(move |a, b| cmp(&a.key, &b.key));
        IntoIter::new(entries)
    }

    /// Sort the set's values by their default ordering.
    ///
    /// See [`sort_unstable_by`](Self::sort_unstable_by) for details.
    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.map.sort_unstable_keys()
    }

    /// Sort the set's values in place using the comparison function `cmp`.
    ///
    /// Computes in **O(n log n)** time. The sort is unstable.
    pub fn sort_unstable_by<F>(&mut self, mut cmp: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.map.sort_unstable_by(move |a, _, b, _| cmp(a, b))
    }

    /// Sort the values of the set and return a by-value iterator of
    /// the values with the result.
    pub fn sorted_unstable_by<F>(self, mut cmp: F) -> IntoIter<T>
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        let mut entries = self.into_entries();
        entries
            .make_contiguous()
            .sort_unstable_by(move |a, b| cmp(&a.key, &b.key));
        IntoIter::new(entries)
    }

    /// Sort the set’s values in place using a key extraction function.
    ///
    /// During sorting, the function is called at most once per entry, by using temporary storage
    /// to remember the results of its evaluation. The order of calls to the function is
    /// unspecified and may change between versions of `indexmap` or the standard library.
    ///
    /// Computes in **O(m n + n log n + c)** time () and **O(n)** space, where the function is
    /// **O(m)**, *n* is the length of the map, and *c* the capacity. The sort is stable.
    pub fn sort_by_cached_key<K, F>(&mut self, mut sort_key: F)
    where
        K: Ord,
        F: FnMut(&T) -> K,
    {
        self.with_contiguous_entries(move |entries| {
            entries.sort_by_cached_key(move |a| sort_key(&a.key));
        });
    }

    /// Search over a sorted set for a value.
    ///
    /// Returns the position where that value is present, or the position where it can be inserted
    /// to maintain the sort. See [`slice::binary_search`] for more details.
    ///
    /// Computes in **O(log(n))** time, which is notably less scalable than looking the value up
    /// using [`get_index_of`][RingSet::get_index_of], but this can also position missing values.
    pub fn binary_search(&self, x: &T) -> Result<usize, usize>
    where
        T: Ord,
    {
        self.map.binary_search_keys(x)
    }

    /// Search over a sorted set with a comparator function.
    ///
    /// Returns the position where that value is present, or the position where it can be inserted
    /// to maintain the sort. See [`slice::binary_search_by`] for more details.
    ///
    /// Computes in **O(log(n))** time.
    #[inline]
    pub fn binary_search_by<'a, F>(&'a self, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&'a T) -> Ordering,
    {
        self.map.binary_search_by(move |key, ()| f(key))
    }

    /// Search over a sorted set with an extraction function.
    ///
    /// Returns the position where that value is present, or the position where it can be inserted
    /// to maintain the sort. See [`slice::binary_search_by_key`] for more details.
    ///
    /// Computes in **O(log(n))** time.
    #[inline]
    pub fn binary_search_by_key<'a, B, F>(&'a self, b: &B, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&'a T) -> B,
        B: Ord,
    {
        self.map.binary_search_by_key(b, move |key, ()| f(key))
    }

    /// Returns the index of the partition point of a sorted set according to the given predicate
    /// (the index of the first element of the second partition).
    ///
    /// See [`slice::partition_point`] for more details.
    ///
    /// Computes in **O(log(n))** time.
    #[must_use]
    pub fn partition_point<P>(&self, mut pred: P) -> usize
    where
        P: FnMut(&T) -> bool,
    {
        self.map.partition_point(move |key, ()| pred(key))
    }

    /// Reverses the order of the set’s values in place.
    ///
    /// Computes in **O(n)** time and **O(1)** space.
    pub fn reverse(&mut self) {
        self.map.reverse()
    }

    /// Returns head and tail slices of all the values in the set.
    ///
    /// Computes in **O(1)** time.
    pub fn as_slices(&self) -> (&Slice<T>, &Slice<T>) {
        let (head, tail) = self.as_entries().as_slices();
        (Slice::from_slice(head), Slice::from_slice(tail))
    }

    /// Rearranges the internal storage of this map so it is one contiguous slice,
    /// which is then returned.
    pub fn make_contiguous(&mut self) -> &Slice<T> {
        Slice::from_slice(self.as_entries_mut().make_contiguous())
    }

    /// Converts into a boxed slice of all the values in the set.
    ///
    /// Note that this will drop the inner hash table and any excess capacity,
    /// and may need to move items if they're not at the beginning of the allocation.
    pub fn into_boxed_slice(self) -> Box<Slice<T>> {
        let entries = Vec::from(self.into_entries());
        Slice::from_boxed(entries.into_boxed_slice())
    }

    /// Get a value by index
    ///
    /// Valid indices are `0 <= index < self.len()`.
    ///
    /// Computes in **O(1)** time.
    pub fn get_index(&self, index: usize) -> Option<&T> {
        self.as_entries().get(index).map(Bucket::key_ref)
    }

    /// Get the first value
    ///
    /// Computes in **O(1)** time.
    #[doc(alias = "first")] // like `BTreeSet`
    pub fn front(&self) -> Option<&T> {
        self.as_entries().get(0).map(Bucket::key_ref)
    }

    /// Get the last value
    ///
    /// Computes in **O(1)** time.
    #[doc(alias = "last")] // like `BTreeSet`
    pub fn back(&self) -> Option<&T> {
        let i = self.len().checked_sub(1)?;
        self.as_entries().get(i).map(Bucket::key_ref)
    }

    /// Remove the value by index
    ///
    /// Valid indices are `0 <= index < self.len()`.
    ///
    /// Like [`VecDeque::swap_remove_back`], the value is removed by swapping it with the
    /// last element of the set and popping it off. **This perturbs
    /// the position of what used to be the last element!**
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_remove_back_index(&mut self, index: usize) -> Option<T> {
        self.map.swap_remove_back_index(index).map(|(x, ())| x)
    }

    /// Remove the value by index
    ///
    /// Valid indices are `0 <= index < self.len()`.
    ///
    /// Like [`VecDeque::swap_remove_front`], the value is removed by swapping it with the
    /// front element of the set and popping it off. **This perturbs
    /// the position of what used to be the front element!**
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_remove_front_index(&mut self, index: usize) -> Option<T> {
        self.map.swap_remove_front_index(index).map(|(x, ())| x)
    }

    /// Remove the value by index
    ///
    /// Valid indices are `0 <= index < self.len()`.
    ///
    /// Like [`VecDeque::remove`], the value is removed by shifting all of the
    /// elements either before or after it, preserving their relative order.
    /// **This perturbs the index of all of the following elements!**
    ///
    /// Computes in **O(n)** time (average).
    pub fn remove_index(&mut self, index: usize) -> Option<T> {
        self.map.remove_index(index).map(|(x, ())| x)
    }

    /// Moves the position of a value from one index to another
    /// by shifting all other values in-between.
    ///
    /// * If `from < to`, the other values will shift down while the targeted value moves up.
    /// * If `from > to`, the other values will shift up while the targeted value moves down.
    ///
    /// ***Panics*** if `from` or `to` are out of bounds.
    ///
    /// Computes in **O(n)** time (average).
    #[track_caller]
    pub fn move_index(&mut self, from: usize, to: usize) {
        self.map.move_index(from, to)
    }

    /// Swaps the position of two values in the set.
    ///
    /// ***Panics*** if `a` or `b` are out of bounds.
    ///
    /// Computes in **O(1)** time (average).
    #[track_caller]
    pub fn swap_indices(&mut self, a: usize, b: usize) {
        self.map.swap_indices(a, b)
    }
}

/// Access [`RingSet`] values at indexed positions.
///
/// # Examples
///
/// ```
/// use ringmap::RingSet;
///
/// let mut set = RingSet::new();
/// for word in "Lorem ipsum dolor sit amet".split_whitespace() {
///     set.insert(word.to_string());
/// }
/// assert_eq!(set[0], "Lorem");
/// assert_eq!(set[1], "ipsum");
/// set.reverse();
/// assert_eq!(set[0], "amet");
/// assert_eq!(set[1], "sit");
/// set.sort();
/// assert_eq!(set[0], "Lorem");
/// assert_eq!(set[1], "amet");
/// ```
///
/// ```should_panic
/// use ringmap::RingSet;
///
/// let mut set = RingSet::new();
/// set.insert("foo");
/// println!("{:?}", set[10]); // panics!
/// ```
impl<T, S> Index<usize> for RingSet<T, S> {
    type Output = T;

    /// Returns a reference to the value at the supplied `index`.
    ///
    /// ***Panics*** if `index` is out of bounds.
    fn index(&self, index: usize) -> &T {
        self.get_index(index).unwrap_or_else(|| {
            panic!(
                "index out of bounds: the len is {len} but the index is {index}",
                len = self.len()
            );
        })
    }
}

impl<T, S> FromIterator<T> for RingSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item = T>>(iterable: I) -> Self {
        let iter = iterable.into_iter().map(|x| (x, ()));
        RingSet {
            map: RingMap::from_iter(iter),
        }
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<T, const N: usize> From<[T; N]> for RingSet<T, RandomState>
where
    T: Eq + Hash,
{
    /// # Examples
    ///
    /// ```
    /// use ringmap::RingSet;
    ///
    /// let set1 = RingSet::from([1, 2, 3, 4]);
    /// let set2: RingSet<_> = [1, 2, 3, 4].into();
    /// assert_eq!(set1, set2);
    /// ```
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<T, S> Extend<T> for RingSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iterable: I) {
        let iter = iterable.into_iter().map(|x| (x, ()));
        self.map.extend(iter);
    }
}

impl<'a, T, S> Extend<&'a T> for RingSet<T, S>
where
    T: Hash + Eq + Copy + 'a,
    S: BuildHasher,
{
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iterable: I) {
        let iter = iterable.into_iter().copied();
        self.extend(iter);
    }
}

impl<T, S> Default for RingSet<T, S>
where
    S: Default,
{
    /// Return an empty [`RingSet`]
    fn default() -> Self {
        RingSet {
            map: RingMap::default(),
        }
    }
}

impl<T, S1, S2> PartialEq<RingSet<T, S2>> for RingSet<T, S1>
where
    T: PartialEq,
{
    fn eq(&self, other: &RingSet<T, S2>) -> bool {
        self.len() == other.len() && self.iter().eq(other)
    }
}

impl<T, S> Eq for RingSet<T, S> where T: Eq {}

impl<T, S1, S2> PartialOrd<RingSet<T, S2>> for RingSet<T, S1>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &RingSet<T, S2>) -> Option<Ordering> {
        self.iter().partial_cmp(other)
    }
}

impl<T, S> Ord for RingSet<T, S>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other)
    }
}

impl<T, S> Hash for RingSet<T, S>
where
    T: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for value in self {
            value.hash(state);
        }
    }
}

impl<T, S> RingSet<T, S>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    /// Returns `true` if `self` has no elements in common with `other`.
    pub fn is_disjoint<S2>(&self, other: &RingSet<T, S2>) -> bool
    where
        S2: BuildHasher,
    {
        if self.len() <= other.len() {
            self.iter().all(move |value| !other.contains(value))
        } else {
            other.iter().all(move |value| !self.contains(value))
        }
    }

    /// Returns `true` if all elements of `self` are contained in `other`.
    pub fn is_subset<S2>(&self, other: &RingSet<T, S2>) -> bool
    where
        S2: BuildHasher,
    {
        self.len() <= other.len() && self.iter().all(move |value| other.contains(value))
    }

    /// Returns `true` if all elements of `other` are contained in `self`.
    pub fn is_superset<S2>(&self, other: &RingSet<T, S2>) -> bool
    where
        S2: BuildHasher,
    {
        other.is_subset(self)
    }

    /// Returns `true` if `self` and `other` contain exactly the same elements,
    /// even if they are not in the same order.
    ///
    /// (Note that `PartialEq for RingSet` **does** consider the order.)
    pub fn set_eq<S2>(&self, other: &RingSet<T, S2>) -> bool
    where
        S2: BuildHasher,
    {
        self.len() == other.len() && self.is_subset(other)
    }
}

impl<T, S1, S2> BitAnd<&RingSet<T, S2>> for &RingSet<T, S1>
where
    T: Eq + Hash + Clone,
    S1: BuildHasher + Default,
    S2: BuildHasher,
{
    type Output = RingSet<T, S1>;

    /// Returns the set intersection, cloned into a new set.
    ///
    /// Values are collected in the same order that they appear in `self`.
    fn bitand(self, other: &RingSet<T, S2>) -> Self::Output {
        self.intersection(other).cloned().collect()
    }
}

impl<T, S1, S2> BitOr<&RingSet<T, S2>> for &RingSet<T, S1>
where
    T: Eq + Hash + Clone,
    S1: BuildHasher + Default,
    S2: BuildHasher,
{
    type Output = RingSet<T, S1>;

    /// Returns the set union, cloned into a new set.
    ///
    /// Values from `self` are collected in their original order, followed by
    /// values that are unique to `other` in their original order.
    fn bitor(self, other: &RingSet<T, S2>) -> Self::Output {
        self.union(other).cloned().collect()
    }
}

impl<T, S1, S2> BitXor<&RingSet<T, S2>> for &RingSet<T, S1>
where
    T: Eq + Hash + Clone,
    S1: BuildHasher + Default,
    S2: BuildHasher,
{
    type Output = RingSet<T, S1>;

    /// Returns the set symmetric-difference, cloned into a new set.
    ///
    /// Values from `self` are collected in their original order, followed by
    /// values from `other` in their original order.
    fn bitxor(self, other: &RingSet<T, S2>) -> Self::Output {
        self.symmetric_difference(other).cloned().collect()
    }
}

impl<T, S1, S2> Sub<&RingSet<T, S2>> for &RingSet<T, S1>
where
    T: Eq + Hash + Clone,
    S1: BuildHasher + Default,
    S2: BuildHasher,
{
    type Output = RingSet<T, S1>;

    /// Returns the set difference, cloned into a new set.
    ///
    /// Values are collected in the same order that they appear in `self`.
    fn sub(self, other: &RingSet<T, S2>) -> Self::Output {
        self.difference(other).cloned().collect()
    }
}
