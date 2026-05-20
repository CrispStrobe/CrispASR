use crispasr::agglomerative_cluster;

fn main() {
    let a = [1.0f32, 0., 0., 0., 0., 0., 0., 0.];
    let b = [0.0f32, 1., 0., 0., 0., 0., 0., 0.];
    let mut embs: Vec<f32> = Vec::new();
    for _ in 0..3 {
        embs.extend_from_slice(&a);
    }
    for _ in 0..3 {
        embs.extend_from_slice(&b);
    }
    let labels = agglomerative_cluster(&embs, 6, 8, 0.5, 8).expect("clustering ok");
    println!("labels = {:?}", labels);
    let mut distinct: std::collections::HashSet<i32> = labels.iter().copied().collect();
    distinct.remove(&-1);
    assert_eq!(distinct.len(), 2, "expected 2 clusters");
    println!("-> 2 distinct clusters, OK");
}
