#[test]
fn test_create_map() {
    let _m = ringmap::ringmap! {
        1 => 2,
        7 => 1,
        2 => 2,
        3 => 3,
    };
}

#[test]
fn test_create_set() {
    let _s = ringmap::ringset! {
        1,
        7,
        2,
        3,
    };
}
