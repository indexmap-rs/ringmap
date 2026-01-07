use super::{Bucket, Core};
use crate::inner::entry::{OccupiedEntry, VacantEntry};
use core::{fmt, mem};

/// Entry for an existing key-value pair in a [`RingMap`][crate::RingMap]
/// or a vacant location to insert one.
pub enum Entry<'a, K, V> {
    /// Existing slot with equivalent key.
    Occupied(OccupiedEntry<'a, K, V>),
    /// Vacant slot (no equivalent key in the map).
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K, V> Entry<'a, K, V> {
    /// Return the index where the key-value pair exists or may be appended.
    ///
    /// Note that some methods may instead prepend new items at index 0.
    pub fn index(&self) -> usize {
        match self {
            Entry::Occupied(entry) => entry.index(),
            Entry::Vacant(entry) => entry.index(),
        }
    }

    #[deprecated = "use `push_back_entry` or `push_front_entry` instead"]
    pub fn insert_entry(self, value: V) -> OccupiedEntry<'a, K, V> {
        self.push_back_entry(value)
    }

    /// Sets the value of the entry (after appending if vacant), and returns an `OccupiedEntry`.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn push_back_entry(self, value: V) -> OccupiedEntry<'a, K, V> {
        match self {
            Entry::Occupied(mut entry) => {
                entry.insert(value);
                entry
            }
            Entry::Vacant(entry) => entry.push_back_entry(value),
        }
    }

    /// Sets the value of the entry (after prepending if vacant), and returns an `OccupiedEntry`.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn push_front_entry(self, value: V) -> OccupiedEntry<'a, K, V> {
        match self {
            Entry::Occupied(mut entry) => {
                entry.insert(value);
                entry
            }
            Entry::Vacant(entry) => entry.push_front_entry(value),
        }
    }

    #[deprecated = "use `or_push_back` or `or_push_front` instead"]
    pub fn or_insert(self, default: V) -> &'a mut V {
        self.or_push_back(default)
    }

    /// Appends the given default value in the entry if it is vacant and returns a mutable
    /// reference to it. Otherwise a mutable reference to an already existent value is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_push_back(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.push_back(default),
        }
    }

    /// Prepends the given default value in the entry if it is vacant and returns a mutable
    /// reference to it. Otherwise a mutable reference to an already existent value is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_push_front(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.push_front(default),
        }
    }

    #[deprecated = "use `or_push_back_default` or `or_push_front_default` instead"]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_push_back_default()
    }

    /// Appends a default-constructed value in the entry if it is vacant and returns a mutable
    /// reference to it. Otherwise a mutable reference to an already existent value is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_push_back_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_push_back_with(V::default)
    }

    /// Prepends a default-constructed value in the entry if it is vacant and returns a mutable
    /// reference to it. Otherwise a mutable reference to an already existent value is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_push_front_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_push_front_with(V::default)
    }

    #[deprecated = "use `or_push_back_with` or `or_push_front_with` instead"]
    pub fn or_insert_with<F>(self, call: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        self.or_push_back_with(call)
    }

    /// Appends the result of the `call` function in the entry if it is vacant and returns a mutable
    /// reference to it. Otherwise a mutable reference to an already existent value is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_push_back_with<F>(self, call: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.push_back(call()),
        }
    }

    /// Prepends the result of the `call` function in the entry if it is vacant and returns a mutable
    /// reference to it. Otherwise a mutable reference to an already existent value is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_push_front_with<F>(self, call: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.push_front(call()),
        }
    }

    #[deprecated = "use `or_push_back_with_key` or `or_push_front_with_key` instead"]
    pub fn or_insert_with_key<F>(self, call: F) -> &'a mut V
    where
        F: FnOnce(&K) -> V,
    {
        self.or_push_back_with_key(call)
    }

    /// Appends the result of the `call` function with a reference to the entry's key if it is
    /// vacant, and returns a mutable reference to the new value. Otherwise a mutable reference to
    /// an already existent value is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_push_back_with_key<F>(self, call: F) -> &'a mut V
    where
        F: FnOnce(&K) -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let value = call(entry.key());
                entry.push_back(value)
            }
        }
    }

    /// Prepends the result of the `call` function with a reference to the entry's key if it is
    /// vacant, and returns a mutable reference to the new value. Otherwise a mutable reference to
    /// an already existent value is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn or_push_front_with_key<F>(self, call: F) -> &'a mut V
    where
        F: FnOnce(&K) -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let value = call(entry.key());
                entry.push_front(value)
            }
        }
    }

    /// Gets a reference to the entry's key, either within the map if occupied,
    /// or else the new key that was used to find the entry.
    pub fn key(&self) -> &K {
        match *self {
            Entry::Occupied(ref entry) => entry.key(),
            Entry::Vacant(ref entry) => entry.key(),
        }
    }

    /// Modifies the entry if it is occupied.
    pub fn and_modify<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        if let Entry::Occupied(entry) = &mut self {
            f(entry.get_mut());
        }
        self
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Entry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut tuple = f.debug_tuple("Entry");
        match self {
            Entry::Vacant(v) => tuple.field(v),
            Entry::Occupied(o) => tuple.field(o),
        };
        tuple.finish()
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for OccupiedEntry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", self.key())
            .field("value", self.get())
            .finish()
    }
}

impl<K: fmt::Debug, V> fmt::Debug for VacantEntry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VacantEntry").field(self.key()).finish()
    }
}

/// A view into an occupied entry in a [`RingMap`][crate::RingMap] obtained by index.
///
/// This `struct` is created from the [`get_index_entry`][crate::RingMap::get_index_entry] method.
pub struct IndexedEntry<'a, K, V> {
    map: &'a mut Core<K, V>,
    // We have a mutable reference to the map, which keeps the index
    // valid and pointing to the correct entry.
    index: usize,
}

impl<'a, K, V> IndexedEntry<'a, K, V> {
    pub(crate) fn new(map: &'a mut Core<K, V>, index: usize) -> Option<Self> {
        if index < map.len() {
            Some(Self { map, index })
        } else {
            None
        }
    }

    /// Return the index of the key-value pair
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    pub(crate) fn into_core(self) -> &'a mut Core<K, V> {
        self.map
    }

    fn get_bucket(&self) -> &Bucket<K, V> {
        &self.map.as_entries()[self.index]
    }

    fn get_bucket_mut(&mut self) -> &mut Bucket<K, V> {
        &mut self.map.as_entries_mut()[self.index]
    }

    fn into_bucket(self) -> &'a mut Bucket<K, V> {
        &mut self.map.as_entries_mut()[self.index]
    }

    /// Gets a reference to the entry's key in the map.
    pub fn key(&self) -> &K {
        &self.get_bucket().key
    }

    pub(crate) fn key_mut(&mut self) -> &mut K {
        &mut self.get_bucket_mut().key
    }

    /// Gets a reference to the entry's value in the map.
    pub fn get(&self) -> &V {
        &self.get_bucket().value
    }

    /// Gets a mutable reference to the entry's value in the map.
    ///
    /// If you need a reference which may outlive the destruction of the
    /// `IndexedEntry` value, see [`into_mut`][Self::into_mut].
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.get_bucket_mut().value
    }

    /// Sets the value of the entry to `value`, and returns the entry's old value.
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Converts into a mutable reference to the entry's value in the map,
    /// with a lifetime bound to the map itself.
    pub fn into_mut(self) -> &'a mut V {
        &mut self.into_bucket().value
    }

    /// Remove and return the key, value pair stored in the map for this entry
    ///
    /// Like [`VecDeque::remove`][super::VecDeque::remove], the pair is removed by shifting all of the
    /// elements either before or after it, preserving their relative order.
    /// **This perturbs the index of all of the following elements!**
    ///
    /// Computes in **O(n)** time (average).
    pub fn remove_entry(self) -> (K, V) {
        self.map.shift_remove_index(self.index).unwrap()
    }

    /// Remove the key, value pair stored in the map for this entry, and return the value.
    ///
    /// Like [`VecDeque::remove`][super::VecDeque::remove], the pair is removed by shifting all of the
    /// elements either before or after it, preserving their relative order.
    /// **This perturbs the index of all of the following elements!**
    ///
    /// Computes in **O(n)** time (average).
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Remove and return the key, value pair stored in the map for this entry
    ///
    /// Like [`VecDeque::swap_remove_back`][super::VecDeque::swap_remove_back], the pair is removed
    /// by swapping it with the last element of the map and popping it off.
    /// **This perturbs the position of what used to be the last element!**
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_remove_back_entry(self) -> (K, V) {
        self.map.swap_remove_back_index(self.index).unwrap()
    }

    /// Remove the key, value pair stored in the map for this entry, and return the value.
    ///
    /// Like [`VecDeque::swap_remove_back`][super::VecDeque::swap_remove_back], the pair is removed
    /// by swapping it with the last element of the map and popping it off.
    /// **This perturbs the position of what used to be the last element!**
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_remove_back(self) -> V {
        self.swap_remove_back_entry().1
    }

    /// Remove and return the key, value pair stored in the map for this entry
    ///
    /// Like [`VecDeque::swap_remove_front`][super::VecDeque::swap_remove_front], the pair is removed
    /// by swapping it with the front element of the map and popping it off.
    /// **This perturbs the position of what used to be the front element!**
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_remove_front_entry(self) -> (K, V) {
        self.map.swap_remove_front_index(self.index).unwrap()
    }

    /// Remove the key, value pair stored in the map for this entry, and return the value.
    ///
    /// Like [`VecDeque::swap_remove_front`][super::VecDeque::swap_remove_front], the pair is removed
    /// by swapping it with the front element of the map and popping it off.
    /// **This perturbs the position of what used to be the front element!**
    ///
    /// Computes in **O(1)** time (average).
    pub fn swap_remove_front(self) -> V {
        self.swap_remove_front_entry().1
    }

    /// Moves the position of the entry to a new index
    /// by shifting all other entries in-between.
    ///
    /// This is equivalent to [`RingMap::move_index`][`crate::RingMap::move_index`]
    /// coming `from` the current [`.index()`][Self::index].
    ///
    /// * If `self.index() < to`, the other pairs will shift down while the targeted pair moves up.
    /// * If `self.index() > to`, the other pairs will shift up while the targeted pair moves down.
    ///
    /// ***Panics*** if `to` is out of bounds.
    ///
    /// Computes in **O(n)** time (average).
    #[track_caller]
    pub fn move_index(self, to: usize) {
        self.map.move_index(self.index, to);
    }

    /// Swaps the position of entry with another.
    ///
    /// This is equivalent to [`RingMap::swap_indices`][`crate::RingMap::swap_indices`]
    /// with the current [`.index()`][Self::index] as one of the two being swapped.
    ///
    /// ***Panics*** if the `other` index is out of bounds.
    ///
    /// Computes in **O(1)** time (average).
    #[track_caller]
    pub fn swap_indices(self, other: usize) {
        self.map.swap_indices(self.index, other);
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for IndexedEntry<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexedEntry")
            .field("index", &self.index)
            .field("key", self.key())
            .field("value", self.get())
            .finish()
    }
}

impl<'a, K, V> From<OccupiedEntry<'a, K, V>> for IndexedEntry<'a, K, V> {
    fn from(other: OccupiedEntry<'a, K, V>) -> Self {
        Self {
            index: other.index(),
            map: other.into_core(),
        }
    }
}

#[test]
fn assert_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Entry<'_, i32, i32>>();
    assert_send_sync::<IndexedEntry<'_, i32, i32>>();
}
