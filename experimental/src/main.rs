use experimental::SumTree;


fn main() {
    let mut tree = SumTree::from_vec(vec![4.2, 1.2, 4.7, 9.9, 8.1]);
    println!("{:?}", tree);
    tree.update(3, 1.1);
    println!("{:?}", tree);
    let sampled = tree.sample_idx(10);
    println!("{:?}", sampled);
    tree.is_correct();
}
