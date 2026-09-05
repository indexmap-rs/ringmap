/// Create a [`RingMap`][crate::RingMap] from a list of key-value pairs
/// and a [`BuildHasherDefault`][core::hash::BuildHasherDefault]-wrapped custom hasher.
///
/// ## Example
///
/// ```
/// use ringmap::ringmap_with_default;
/// use fnv::FnvHasher;
///
/// let map = ringmap_with_default!{
///     FnvHasher;
///     "a" => 1,
///     "b" => 2,
/// };
/// assert_eq!(map["a"], 1);
/// assert_eq!(map["b"], 2);
/// assert_eq!(map.get("c"), None);
///
/// // "a" is the first key
/// assert_eq!(map.keys().next(), Some(&"a"));
/// ```
///
/// This can also be initialized in `const` contexts:
///
/// ```
/// use ringmap::{RingMap, ringmap_with_default};
/// use fnv::FnvBuildHasher; // = BuildHasherDefault<FnvHasher>
/// use std::sync::Mutex;
///
/// static GLOBAL: Mutex<RingMap<String, i32, FnvBuildHasher>> =
///     Mutex::new(ringmap_with_default!());
///
/// if let Ok(mut map) = GLOBAL.lock() {
///     map.insert("a".into(), 1);
///     map.insert("b".into(), 2);
/// }
///
/// assert_eq!(GLOBAL.lock().unwrap()["a"], 1);
/// assert_eq!(GLOBAL.lock().unwrap()["b"], 2);
/// ```
#[macro_export]
macro_rules! ringmap_with_default {
    () => { const {
        $crate::RingMap::with_hasher(
            // Let type inference figure out the hasher:
            ::core::hash::BuildHasherDefault::new(),
        )
    }};
    ($H:ty $(;)?) => { const {
        $crate::RingMap::with_hasher(
            // Specify your custom `H` (must implement Default + Hasher) as the hasher:
            ::core::hash::BuildHasherDefault::<$H>::new(),
        )
    }};
    ($H:ty; $($key:expr => $value:expr),+ $(,)?) => {{
        let mut map = $crate::RingMap::with_capacity_and_hasher(
            // Note: `stringify!($key)` is just here to consume the repetition,
            // but we throw away that string literal during constant evaluation.
            const { <[()]>::len(&[$({ stringify!($key); }),*]) },
            // Specify your custom `H` (must implement Default + Hasher) as the hasher:
            ::core::hash::BuildHasherDefault::<$H>::new(),
        );
        $(
            map.insert($key, $value);
        )+
        map
    }};
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
#[macro_export]
/// Create a [`RingMap`][crate::RingMap] from a list of key-value pairs
///
/// ## Example
///
/// ```
/// use ringmap::ringmap;
///
/// let map = ringmap!{
///     "a" => 1,
///     "b" => 2,
/// };
/// assert_eq!(map["a"], 1);
/// assert_eq!(map["b"], 2);
/// assert_eq!(map.get("c"), None);
///
/// // "a" is the first key
/// assert_eq!(map.keys().next(), Some(&"a"));
/// ```
macro_rules! ringmap {
    () => { $crate::RingMap::new() };
    ($($key:expr => $value:expr),+ $(,)?) => {{
        let mut map = $crate::RingMap::with_capacity(
            // Note: `stringify!($key)` is just here to consume the repetition,
            // but we throw away that string literal during constant evaluation.
            const { <[()]>::len(&[$({ stringify!($key); }),*]) },
        );
        $(
            map.insert($key, $value);
        )+
        map
    }};
}

/// Create a [`RingSet`][crate::RingSet] from a list of values
/// and a [`BuildHasherDefault`][core::hash::BuildHasherDefault]-wrapped custom hasher.
///
/// ## Example
///
/// ```
/// use ringmap::ringset_with_default;
/// use fnv::FnvHasher;
///
/// let set = ringset_with_default!{
///     FnvHasher;
///     "a",
///     "b",
/// };
/// assert!(set.contains("a"));
/// assert!(set.contains("b"));
/// assert!(!set.contains("c"));
///
/// // "a" is the first value
/// assert_eq!(set.iter().next(), Some(&"a"));
/// ```
///
/// This can also be initialized in `const` contexts:
///
/// ```
/// use ringmap::{RingSet, ringset_with_default};
/// use fnv::FnvBuildHasher; // = BuildHasherDefault<FnvHasher>
/// use std::sync::Mutex;
///
/// static INTERN: Mutex<RingSet<String, FnvBuildHasher>> =
///     Mutex::new(ringset_with_default!());
///
/// if let Ok(mut set) = INTERN.lock() {
///     set.insert("a".into());
///     set.insert("b".into());
///     set.insert("c".into());
/// }
///
/// assert!(INTERN.lock().unwrap().contains("a"));
/// assert!(INTERN.lock().unwrap().contains("b"));
/// assert!(INTERN.lock().unwrap().contains("c"));
/// ```
#[macro_export]
macro_rules! ringset_with_default {
    () => { const {
        $crate::RingSet::with_hasher(
            // Let type inference figure out the hasher:
            ::core::hash::BuildHasherDefault::new(),
        )
    }};
    ($H:ty $(;)?) => { const {
        $crate::RingSet::with_hasher(
            // Specify your custom `H` (must implement Default + Hasher) as the hasher:
            ::core::hash::BuildHasherDefault::<$H>::new(),
        )
    }};
    ($H:ty; $($value:expr),+ $(,)?) => {{
        let mut set = $crate::RingSet::with_capacity_and_hasher(
            // Note: `stringify!($value)` is just here to consume the repetition,
            // but we throw away that string literal during constant evaluation.
            const { <[()]>::len(&[$({ stringify!($value); }),*]) },
            // Specify your custom `H` (must implement Default + Hasher) as the hasher:
            ::core::hash::BuildHasherDefault::<$H>::new(),
        );
        $(
            set.insert($value);
        )+
        set
    }};
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
#[macro_export]
/// Create a [`RingSet`][crate::RingSet] from a list of values
///
/// ## Example
///
/// ```
/// use ringmap::ringset;
///
/// let set = ringset!{
///     "a",
///     "b",
/// };
/// assert!(set.contains("a"));
/// assert!(set.contains("b"));
/// assert!(!set.contains("c"));
///
/// // "a" is the first value
/// assert_eq!(set.iter().next(), Some(&"a"));
/// ```
macro_rules! ringset {
    () => { $crate::RingSet::new() };
    ($($value:expr),+ $(,)?) => {{
        let mut set = $crate::RingSet::with_capacity(
            // Note: `stringify!($value)` is just here to consume the repetition,
            // but we throw away that string literal during constant evaluation.
            const { <[()]>::len(&[$({ stringify!($value); }),*]) },
        );
        $(
            set.insert($value);
        )+
        set
    }};
}

// generate all the Iterator methods by just forwarding to the underlying
// self.iter and mapping its element.
macro_rules! iterator_methods {
    // $map_elt is the mapping function from the underlying iterator's element
    // same mapping function for both options and iterators
    ($map_elt:expr) => {
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next().map($map_elt)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.iter.size_hint()
        }

        fn count(self) -> usize {
            self.iter.len()
        }

        fn nth(&mut self, n: usize) -> Option<Self::Item> {
            self.iter.nth(n).map($map_elt)
        }

        fn last(mut self) -> Option<Self::Item> {
            self.next_back()
        }

        fn fold<Acc, F>(self, acc: Acc, f: F) -> Acc
        where
            F: FnMut(Acc, Self::Item) -> Acc,
        {
            self.iter.map($map_elt).fold(acc, f)
        }

        fn collect<C>(self) -> C
        where
            C: FromIterator<Self::Item>,
        {
            // NB: forwarding this directly to standard iterators will
            // allow it to leverage unstable traits like `TrustedLen`.
            self.iter.map($map_elt).collect()
        }
    };
}

macro_rules! double_ended_iterator_methods {
    // $map_elt is the mapping function from the underlying iterator's element
    // same mapping function for both options and iterators
    ($map_elt:expr) => {
        fn next_back(&mut self) -> Option<Self::Item> {
            self.iter.next_back().map($map_elt)
        }

        fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
            self.iter.nth_back(n).map($map_elt)
        }

        fn rfold<Acc, F>(self, acc: Acc, f: F) -> Acc
        where
            F: FnMut(Acc, Self::Item) -> Acc,
        {
            self.iter.map($map_elt).rfold(acc, f)
        }
    };
}

// generate `ParallelIterator` methods by just forwarding to the underlying
// self.entries and mapping its elements.
#[cfg(feature = "rayon")]
macro_rules! parallel_iterator_methods {
    // $map_elt is the mapping function from the underlying iterator's element
    ($map_elt:expr) => {
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            self.entries
                .into_par_iter()
                .map($map_elt)
                .drive_unindexed(consumer)
        }

        // NB: This allows indexed collection, e.g. directly into a `Vec`, but the
        // underlying iterator must really be indexed.  We should remove this if we
        // start having tombstones that must be filtered out.
        fn opt_len(&self) -> Option<usize> {
            Some(self.entries.len())
        }
    };
}

// generate `IndexedParallelIterator` methods by just forwarding to the underlying
// self.entries and mapping its elements.
#[cfg(feature = "rayon")]
macro_rules! indexed_parallel_iterator_methods {
    // $map_elt is the mapping function from the underlying iterator's element
    ($map_elt:expr) => {
        fn drive<C>(self, consumer: C) -> C::Result
        where
            C: Consumer<Self::Item>,
        {
            self.entries.into_par_iter().map($map_elt).drive(consumer)
        }

        fn len(&self) -> usize {
            self.entries.len()
        }

        fn with_producer<CB>(self, callback: CB) -> CB::Output
        where
            CB: ProducerCallback<Self::Item>,
        {
            self.entries
                .into_par_iter()
                .map($map_elt)
                .with_producer(callback)
        }
    };
}
