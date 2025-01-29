use super::core::RingMapCore;
use super::{Bucket, Entries, RingMap};

use alloc::collections::vec_deque::{self, VecDeque};
use core::fmt;
use core::hash::{BuildHasher, Hash};
use core::iter::FusedIterator;
use core::ops::{Index, RangeBounds};
use core::slice;

impl<'a, K, V, S> IntoIterator for &'a RingMap<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut RingMap<K, V, S> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V, S> IntoIterator for RingMap<K, V, S> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.into_entries())
    }
}

/// Internal iterator over `VecDeque` slices
pub(crate) struct Buckets<'a, K, V> {
    head: slice::Iter<'a, Bucket<K, V>>,
    tail: slice::Iter<'a, Bucket<K, V>>,
}

impl<'a, K, V> Buckets<'a, K, V> {
    pub(crate) fn new(entries: &'a VecDeque<Bucket<K, V>>) -> Self {
        Self::from_slices(entries.as_slices())
    }

    pub(crate) fn from_slices((head, tail): (&'a [Bucket<K, V>], &'a [Bucket<K, V>])) -> Self {
        Self {
            head: head.iter(),
            tail: tail.iter(),
        }
    }
}

impl<'a, K, V> Iterator for Buckets<'a, K, V> {
    type Item = &'a Bucket<K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.head.next() {
            next @ Some(_) => next,
            None => self.tail.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }

    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        if n < self.head.len() {
            return self.head.nth(n);
        }
        if self.head.len() > 0 {
            n -= self.head.len();
            self.head = [].iter();
        }
        self.tail.nth(n)
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    fn collect<C>(self) -> C
    where
        C: FromIterator<Self::Item>,
    {
        self.head.chain(self.tail).collect()
    }
}

impl<K, V> DoubleEndedIterator for Buckets<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.tail.next_back() {
            next @ Some(_) => next,
            None => self.head.next_back(),
        }
    }

    fn nth_back(&mut self, mut n: usize) -> Option<Self::Item> {
        if n < self.tail.len() {
            return self.tail.nth_back(n);
        }
        if self.tail.len() > 0 {
            n -= self.tail.len();
            self.tail = [].iter();
        }
        self.head.nth_back(n)
    }
}

impl<K, V> ExactSizeIterator for Buckets<'_, K, V> {
    fn len(&self) -> usize {
        self.head.len() + self.tail.len()
    }
}

impl<K, V> Clone for Buckets<'_, K, V> {
    fn clone(&self) -> Self {
        Self {
            head: self.head.clone(),
            tail: self.tail.clone(),
        }
    }
}

impl<K, V> Default for Buckets<'_, K, V> {
    fn default() -> Self {
        Self {
            head: [].iter(),
            tail: [].iter(),
        }
    }
}

/// Internal iterator over `VecDeque` mutable slices
struct BucketsMut<'a, K, V> {
    head: slice::IterMut<'a, Bucket<K, V>>,
    tail: slice::IterMut<'a, Bucket<K, V>>,
}

impl<'a, K, V> BucketsMut<'a, K, V> {
    fn new(entries: &'a mut VecDeque<Bucket<K, V>>) -> Self {
        Self::from_mut_slices(entries.as_mut_slices())
    }

    fn from_mut_slices((head, tail): (&'a mut [Bucket<K, V>], &'a mut [Bucket<K, V>])) -> Self {
        Self {
            head: head.iter_mut(),
            tail: tail.iter_mut(),
        }
    }

    fn iter(&self) -> Buckets<'_, K, V> {
        Buckets::from_slices((self.head.as_slice(), self.tail.as_slice()))
    }
}

impl<'a, K, V> Iterator for BucketsMut<'a, K, V> {
    type Item = &'a mut Bucket<K, V>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.head.next() {
            next @ Some(_) => next,
            None => self.tail.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }

    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        if n < self.head.len() {
            return self.head.nth(n);
        }
        if self.head.len() > 0 {
            n -= self.head.len();
            self.head = [].iter_mut();
        }
        self.tail.nth(n)
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    fn collect<C>(self) -> C
    where
        C: FromIterator<Self::Item>,
    {
        self.head.chain(self.tail).collect()
    }
}

impl<K, V> DoubleEndedIterator for BucketsMut<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.tail.next_back() {
            next @ Some(_) => next,
            None => self.head.next_back(),
        }
    }

    fn nth_back(&mut self, mut n: usize) -> Option<Self::Item> {
        if n < self.tail.len() {
            return self.tail.nth_back(n);
        }
        if self.tail.len() > 0 {
            n -= self.tail.len();
            self.tail = [].iter_mut();
        }
        self.head.nth_back(n)
    }
}

impl<K, V> ExactSizeIterator for BucketsMut<'_, K, V> {
    fn len(&self) -> usize {
        self.head.len() + self.tail.len()
    }
}

impl<K, V> Default for BucketsMut<'_, K, V> {
    fn default() -> Self {
        Self {
            head: [].iter_mut(),
            tail: [].iter_mut(),
        }
    }
}

/// An iterator over the entries of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::iter`] method.
/// See its documentation for more.
pub struct Iter<'a, K, V> {
    iter: Buckets<'a, K, V>,
}

impl<'a, K, V> Iter<'a, K, V> {
    pub(super) fn new(entries: &'a VecDeque<Bucket<K, V>>) -> Self {
        Self {
            iter: Buckets::new(entries),
        }
    }

    pub(super) fn from_slices(slices: (&'a [Bucket<K, V>], &'a [Bucket<K, V>])) -> Self {
        Self {
            iter: Buckets::from_slices(slices),
        }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    iterator_methods!(Bucket::refs);
}

impl<K, V> DoubleEndedIterator for Iter<'_, K, V> {
    double_ended_iterator_methods!(Bucket::refs);
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> FusedIterator for Iter<'_, K, V> {}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
impl<K, V> Clone for Iter<'_, K, V> {
    fn clone(&self) -> Self {
        Iter {
            iter: self.iter.clone(),
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Iter<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<K, V> Default for Iter<'_, K, V> {
    fn default() -> Self {
        Self {
            iter: Default::default(),
        }
    }
}

/// A mutable iterator over the entries of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::iter_mut`] method.
/// See its documentation for more.
pub struct IterMut<'a, K, V> {
    iter: BucketsMut<'a, K, V>,
}

impl<'a, K, V> IterMut<'a, K, V> {
    pub(super) fn new(entries: &'a mut VecDeque<Bucket<K, V>>) -> Self {
        Self {
            iter: BucketsMut::new(entries),
        }
    }

    pub(super) fn from_mut_slices(
        slices: (&'a mut [Bucket<K, V>], &'a mut [Bucket<K, V>]),
    ) -> Self {
        Self {
            iter: BucketsMut::from_mut_slices(slices),
        }
    }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    iterator_methods!(Bucket::ref_mut);
}

impl<K, V> DoubleEndedIterator for IterMut<'_, K, V> {
    double_ended_iterator_methods!(Bucket::ref_mut);
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> FusedIterator for IterMut<'_, K, V> {}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for IterMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.iter.iter().map(Bucket::refs);
        f.debug_list().entries(iter).finish()
    }
}

impl<K, V> Default for IterMut<'_, K, V> {
    fn default() -> Self {
        Self {
            iter: Default::default(),
        }
    }
}

/// A mutable iterator over the entries of an [`RingMap`].
///
/// This `struct` is created by the [`MutableKeys::iter_mut2`][super::MutableKeys::iter_mut2] method.
/// See its documentation for more.
pub struct IterMut2<'a, K, V> {
    iter: BucketsMut<'a, K, V>,
}

impl<'a, K, V> IterMut2<'a, K, V> {
    pub(super) fn new(entries: &'a mut VecDeque<Bucket<K, V>>) -> Self {
        Self {
            iter: BucketsMut::new(entries),
        }
    }
}

impl<'a, K, V> Iterator for IterMut2<'a, K, V> {
    type Item = (&'a mut K, &'a mut V);

    iterator_methods!(Bucket::muts);
}

impl<K, V> DoubleEndedIterator for IterMut2<'_, K, V> {
    double_ended_iterator_methods!(Bucket::muts);
}

impl<K, V> ExactSizeIterator for IterMut2<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> FusedIterator for IterMut2<'_, K, V> {}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for IterMut2<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.iter.iter().map(Bucket::refs);
        f.debug_list().entries(iter).finish()
    }
}

impl<K, V> Default for IterMut2<'_, K, V> {
    fn default() -> Self {
        Self {
            iter: Default::default(),
        }
    }
}

/// An owning iterator over the entries of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::into_iter`] method
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
#[derive(Clone)]
pub struct IntoIter<K, V> {
    iter: vec_deque::IntoIter<Bucket<K, V>>,
}

impl<K, V> IntoIter<K, V> {
    pub(super) fn new(entries: VecDeque<Bucket<K, V>>) -> Self {
        Self {
            iter: entries.into_iter(),
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    iterator_methods!(Bucket::key_value);
}

impl<K, V> DoubleEndedIterator for IntoIter<K, V> {
    double_ended_iterator_methods!(Bucket::key_value);
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> FusedIterator for IntoIter<K, V> {}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for IntoIter<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME
        // let iter = self.iter.as_slice().iter().map(Bucket::refs);
        // f.debug_list().entries(iter).finish()
        f.debug_struct("IntoIter").finish_non_exhaustive()
    }
}

impl<K, V> Default for IntoIter<K, V> {
    fn default() -> Self {
        Self {
            iter: VecDeque::new().into_iter(),
        }
    }
}

/// A draining iterator over the entries of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::drain`] method.
/// See its documentation for more.
pub struct Drain<'a, K, V> {
    iter: vec_deque::Drain<'a, Bucket<K, V>>,
}

impl<'a, K, V> Drain<'a, K, V> {
    pub(super) fn new(iter: vec_deque::Drain<'a, Bucket<K, V>>) -> Self {
        Self { iter }
    }
}

impl<K, V> Iterator for Drain<'_, K, V> {
    type Item = (K, V);

    iterator_methods!(Bucket::key_value);
}

impl<K, V> DoubleEndedIterator for Drain<'_, K, V> {
    double_ended_iterator_methods!(Bucket::key_value);
}

impl<K, V> ExactSizeIterator for Drain<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> FusedIterator for Drain<'_, K, V> {}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Drain<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME
        // let iter = self.iter.as_slice().iter().map(Bucket::refs);
        // f.debug_list().entries(iter).finish()
        f.debug_struct("Drain").finish_non_exhaustive()
    }
}

/// An iterator over the keys of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::keys`] method.
/// See its documentation for more.
pub struct Keys<'a, K, V> {
    iter: Buckets<'a, K, V>,
}

impl<'a, K, V> Keys<'a, K, V> {
    pub(super) fn new(entries: &'a VecDeque<Bucket<K, V>>) -> Self {
        Self {
            iter: Buckets::new(entries),
        }
    }

    pub(super) fn from_slices(slices: (&'a [Bucket<K, V>], &'a [Bucket<K, V>])) -> Self {
        Self {
            iter: Buckets::from_slices(slices),
        }
    }
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    iterator_methods!(Bucket::key_ref);
}

impl<K, V> DoubleEndedIterator for Keys<'_, K, V> {
    double_ended_iterator_methods!(Bucket::key_ref);
}

impl<K, V> ExactSizeIterator for Keys<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> FusedIterator for Keys<'_, K, V> {}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
impl<K, V> Clone for Keys<'_, K, V> {
    fn clone(&self) -> Self {
        Keys {
            iter: self.iter.clone(),
        }
    }
}

impl<K: fmt::Debug, V> fmt::Debug for Keys<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<K, V> Default for Keys<'_, K, V> {
    fn default() -> Self {
        Self {
            iter: Default::default(),
        }
    }
}

/// Access [`RingMap`] keys at indexed positions.
///
/// While [`Index<usize> for RingMap`][values] accesses a map's values,
/// indexing through [`RingMap::keys`] offers an alternative to access a map's
/// keys instead.
///
/// [values]: RingMap#impl-Index<usize>-for-RingMap<K,+V,+S>
///
/// Since `Keys` is also an iterator, consuming items from the iterator will
/// offset the effective indexes. Similarly, if `Keys` is obtained from
/// [`Slice::keys`][super::Slice::keys], indexes will be interpreted relative to the position of
/// that slice.
///
/// # Examples
///
/// ```
/// use ringmap::RingMap;
///
/// let mut map = RingMap::new();
/// for word in "Lorem ipsum dolor sit amet".split_whitespace() {
///     map.insert(word.to_lowercase(), word.to_uppercase());
/// }
///
/// assert_eq!(map[0], "LOREM");
/// assert_eq!(map.keys()[0], "lorem");
/// assert_eq!(map[1], "IPSUM");
/// assert_eq!(map.keys()[1], "ipsum");
///
/// map.reverse();
/// assert_eq!(map.keys()[0], "amet");
/// assert_eq!(map.keys()[1], "sit");
///
/// map.sort_keys();
/// assert_eq!(map.keys()[0], "amet");
/// assert_eq!(map.keys()[1], "dolor");
///
/// // Advancing the iterator will offset the indexing
/// let mut keys = map.keys();
/// assert_eq!(keys[0], "amet");
/// assert_eq!(keys.next().map(|s| &**s), Some("amet"));
/// assert_eq!(keys[0], "dolor");
/// assert_eq!(keys[1], "ipsum");
///
/// // Slices may have an offset as well
/// let (head, tail) = map.as_slices();
/// assert!(tail.is_empty());
/// let slice = &head[2..];
/// assert_eq!(slice[0], "IPSUM");
/// assert_eq!(slice.keys()[0], "ipsum");
/// ```
///
/// ```should_panic
/// use ringmap::RingMap;
///
/// let mut map = RingMap::new();
/// map.insert("foo", 1);
/// println!("{:?}", map.keys()[10]); // panics!
/// ```
impl<K, V> Index<usize> for Keys<'_, K, V> {
    type Output = K;

    /// Returns a reference to the key at the supplied `index`.
    ///
    /// ***Panics*** if `index` is out of bounds.
    fn index(&self, index: usize) -> &K {
        let Buckets { head, tail } = &self.iter;
        if index < head.len() {
            &head.as_slice()[index].key
        } else {
            &tail.as_slice()[index - head.len()].key
        }
    }
}

/// An owning iterator over the keys of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::into_keys`] method.
/// See its documentation for more.
pub struct IntoKeys<K, V> {
    iter: vec_deque::IntoIter<Bucket<K, V>>,
}

impl<K, V> IntoKeys<K, V> {
    pub(super) fn new(entries: VecDeque<Bucket<K, V>>) -> Self {
        Self {
            iter: entries.into_iter(),
        }
    }
}

impl<K, V> Iterator for IntoKeys<K, V> {
    type Item = K;

    iterator_methods!(Bucket::key);
}

impl<K, V> DoubleEndedIterator for IntoKeys<K, V> {
    double_ended_iterator_methods!(Bucket::key);
}

impl<K, V> ExactSizeIterator for IntoKeys<K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> FusedIterator for IntoKeys<K, V> {}

impl<K: fmt::Debug, V> fmt::Debug for IntoKeys<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME
        // let iter = self.iter.as_slice().iter().map(Bucket::key_ref);
        // f.debug_list().entries(iter).finish()
        f.debug_struct("IntoKeys").finish_non_exhaustive()
    }
}

impl<K, V> Default for IntoKeys<K, V> {
    fn default() -> Self {
        Self {
            iter: VecDeque::new().into_iter(),
        }
    }
}

/// An iterator over the values of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::values`] method.
/// See its documentation for more.
pub struct Values<'a, K, V> {
    iter: Buckets<'a, K, V>,
}

impl<'a, K, V> Values<'a, K, V> {
    pub(super) fn new(entries: &'a VecDeque<Bucket<K, V>>) -> Self {
        Self {
            iter: Buckets::new(entries),
        }
    }

    pub(super) fn from_slices(slices: (&'a [Bucket<K, V>], &'a [Bucket<K, V>])) -> Self {
        Self {
            iter: Buckets::from_slices(slices),
        }
    }
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    iterator_methods!(Bucket::value_ref);
}

impl<K, V> DoubleEndedIterator for Values<'_, K, V> {
    double_ended_iterator_methods!(Bucket::value_ref);
}

impl<K, V> ExactSizeIterator for Values<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> FusedIterator for Values<'_, K, V> {}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
impl<K, V> Clone for Values<'_, K, V> {
    fn clone(&self) -> Self {
        Values {
            iter: self.iter.clone(),
        }
    }
}

impl<K, V: fmt::Debug> fmt::Debug for Values<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<K, V> Default for Values<'_, K, V> {
    fn default() -> Self {
        Self {
            iter: Default::default(),
        }
    }
}

/// A mutable iterator over the values of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::values_mut`] method.
/// See its documentation for more.
pub struct ValuesMut<'a, K, V> {
    iter: BucketsMut<'a, K, V>,
}

impl<'a, K, V> ValuesMut<'a, K, V> {
    pub(super) fn new(entries: &'a mut VecDeque<Bucket<K, V>>) -> Self {
        Self {
            iter: BucketsMut::new(entries),
        }
    }

    pub(super) fn from_mut_slices(
        slices: (&'a mut [Bucket<K, V>], &'a mut [Bucket<K, V>]),
    ) -> Self {
        Self {
            iter: BucketsMut::from_mut_slices(slices),
        }
    }
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;

    iterator_methods!(Bucket::value_mut);
}

impl<K, V> DoubleEndedIterator for ValuesMut<'_, K, V> {
    double_ended_iterator_methods!(Bucket::value_mut);
}

impl<K, V> ExactSizeIterator for ValuesMut<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> FusedIterator for ValuesMut<'_, K, V> {}

impl<K, V: fmt::Debug> fmt::Debug for ValuesMut<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let iter = self.iter.iter().map(Bucket::value_ref);
        f.debug_list().entries(iter).finish()
    }
}

impl<K, V> Default for ValuesMut<'_, K, V> {
    fn default() -> Self {
        Self {
            iter: Default::default(),
        }
    }
}

/// An owning iterator over the values of an [`RingMap`].
///
/// This `struct` is created by the [`RingMap::into_values`] method.
/// See its documentation for more.
pub struct IntoValues<K, V> {
    iter: vec_deque::IntoIter<Bucket<K, V>>,
}

impl<K, V> IntoValues<K, V> {
    pub(super) fn new(entries: VecDeque<Bucket<K, V>>) -> Self {
        Self {
            iter: entries.into_iter(),
        }
    }
}

impl<K, V> Iterator for IntoValues<K, V> {
    type Item = V;

    iterator_methods!(Bucket::value);
}

impl<K, V> DoubleEndedIterator for IntoValues<K, V> {
    double_ended_iterator_methods!(Bucket::value);
}

impl<K, V> ExactSizeIterator for IntoValues<K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> FusedIterator for IntoValues<K, V> {}

impl<K, V: fmt::Debug> fmt::Debug for IntoValues<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME
        // let iter = self.iter.as_slice().iter().map(Bucket::value_ref);
        // f.debug_list().entries(iter).finish()
        f.debug_struct("IntoValues").finish_non_exhaustive()
    }
}

impl<K, V> Default for IntoValues<K, V> {
    fn default() -> Self {
        Self {
            iter: VecDeque::new().into_iter(),
        }
    }
}

/// A splicing iterator for `RingMap`.
///
/// This `struct` is created by [`RingMap::splice()`].
/// See its documentation for more.
pub struct Splice<'a, I, K, V, S>
where
    I: Iterator<Item = (K, V)>,
    K: Hash + Eq,
    S: BuildHasher,
{
    map: &'a mut RingMap<K, V, S>,
    tail: RingMapCore<K, V>,
    drain: vec_deque::IntoIter<Bucket<K, V>>,
    replace_with: I,
}

impl<'a, I, K, V, S> Splice<'a, I, K, V, S>
where
    I: Iterator<Item = (K, V)>,
    K: Hash + Eq,
    S: BuildHasher,
{
    #[track_caller]
    pub(super) fn new<R>(map: &'a mut RingMap<K, V, S>, range: R, replace_with: I) -> Self
    where
        R: RangeBounds<usize>,
    {
        let (tail, drain) = map.core.split_splice(range);
        Self {
            map,
            tail,
            drain,
            replace_with,
        }
    }
}

impl<I, K, V, S> Drop for Splice<'_, I, K, V, S>
where
    I: Iterator<Item = (K, V)>,
    K: Hash + Eq,
    S: BuildHasher,
{
    fn drop(&mut self) {
        // Finish draining unconsumed items. We don't strictly *have* to do this
        // manually, since we already split it into separate memory, but it will
        // match the drop order of `vec::Splice` items this way.
        let _ = self.drain.nth(usize::MAX);

        // Now insert all the new items. If a key matches an existing entry, it
        // keeps the original position and only replaces the value, like `insert`.
        while let Some((key, value)) = self.replace_with.next() {
            // Since the tail is disjoint, we can try to update it first,
            // or else insert (update or append) the primary map.
            let hash = self.map.hash(&key);
            if let Some(i) = self.tail.get_index_of(hash, &key) {
                self.tail.as_entries_mut()[i].value = value;
            } else {
                self.map.core.push_back(hash, key, value);
            }
        }

        // Finally, re-append the tail
        self.map.core.append_unchecked(&mut self.tail);
    }
}

impl<I, K, V, S> Iterator for Splice<'_, I, K, V, S>
where
    I: Iterator<Item = (K, V)>,
    K: Hash + Eq,
    S: BuildHasher,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next().map(Bucket::key_value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

impl<I, K, V, S> DoubleEndedIterator for Splice<'_, I, K, V, S>
where
    I: Iterator<Item = (K, V)>,
    K: Hash + Eq,
    S: BuildHasher,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back().map(Bucket::key_value)
    }
}

impl<I, K, V, S> ExactSizeIterator for Splice<'_, I, K, V, S>
where
    I: Iterator<Item = (K, V)>,
    K: Hash + Eq,
    S: BuildHasher,
{
    fn len(&self) -> usize {
        self.drain.len()
    }
}

impl<I, K, V, S> FusedIterator for Splice<'_, I, K, V, S>
where
    I: Iterator<Item = (K, V)>,
    K: Hash + Eq,
    S: BuildHasher,
{
}

impl<I, K, V, S> fmt::Debug for Splice<'_, I, K, V, S>
where
    I: fmt::Debug + Iterator<Item = (K, V)>,
    K: fmt::Debug + Hash + Eq,
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Follow `vec::Splice` in only printing the drain and replacement
        f.debug_struct("Splice")
            .field("drain", &self.drain)
            .field("replace_with", &self.replace_with)
            .finish()
    }
}
