use burn::optim::{AdamConfig, RmsPropConfig};

fn main() {
    let rmsprop = RmsPropConfig::new(/* ... */);
    let adam = AdamConfig::new();

    std::fs::write("rmsprop.json", burn::config::config_to_json(&rmsprop)).expect("fail");
    std::fs::write("adam.json", burn::config::config_to_json(&adam)).expect("fail");
}