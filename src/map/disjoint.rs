#![allow(unsafe_code)]

use crate::GetDisjointMutError;

/// Like `slice::get_disjoint_mut`, although we're not dealing with ranges (yet),
/// and we're also dealing with `VecDeque`'s split across two slices.
pub(super) fn get_disjoint_mut<'a, T, const N: usize>(
    head: &'a mut [T],
    tail: &'a mut [T],
    indices: [usize; N],
) -> Result<[&'a mut T; N], GetDisjointMutError> {
    let mid = head.len();
    let len = mid + tail.len();

    // SAFETY: Can't allow duplicate indices as we would return mutable refs to the same data.
    for i in 0..N {
        let idx = indices[i];
        if idx >= len {
            return Err(GetDisjointMutError::IndexOutOfBounds);
        } else if indices[..i].contains(&idx) {
            return Err(GetDisjointMutError::OverlappingIndices);
        }
    }

    let head_ptr = head.as_mut_ptr();
    let tail_ptr = tail.as_mut_ptr();
    Ok(indices.map(|idx| {
        // SAFETY: The base pointers are valid as they come from slices and the reference is always
        // in-bounds & unique as we've already checked the indices above.
        unsafe {
            let ptr = match idx.checked_sub(mid) {
                None => head_ptr.add(idx),
                Some(tidx) => tail_ptr.add(tidx),
            };
            &mut *ptr
        }
    }))
}

/// Like `slice::get_disjoint_mut` but with optional indices,
/// allowing for absent keys from the user's original request.
#[track_caller]
pub(super) fn get_disjoint_opt_mut<'a, T, const N: usize>(
    head: &'a mut [T],
    tail: &'a mut [T],
    indices: [Option<usize>; N],
) -> [Option<&'a mut T>; N] {
    let mid = head.len();
    let len = mid + tail.len();

    // SAFETY: Can't allow duplicate indices as we would return mutable refs to the same data.
    for i in 0..N {
        if let Some(idx) = indices[i] {
            if idx >= len {
                unreachable!("`get_index_of` returned an out-of-bounds index");
            } else if indices[..i].contains(&Some(idx)) {
                panic!("duplicate keys found");
            }
        }
    }

    let head_ptr = head.as_mut_ptr();
    let tail_ptr = tail.as_mut_ptr();
    indices.map(|idx_opt| {
        match idx_opt {
            Some(idx) => {
                // SAFETY: The base pointers are valid as they come from slices and the reference is always
                // in-bounds & unique as we've already checked the indices above.
                unsafe {
                    let ptr = match idx.checked_sub(mid) {
                        None => head_ptr.add(idx),
                        Some(tidx) => tail_ptr.add(tidx),
                    };
                    Some(&mut *ptr)
                }
            }
            None => None,
        }
    })
}
