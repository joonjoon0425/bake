use pyo3::prelude::*;
use pyo3::types::*;

pub fn display_sys() -> PyResult<()> {
    Python::attach(|py| {
        let sys = py.import("sys")?;
        let version: String = sys.getattr("version")?.extract()?;

        let locals = [("os", py.import("os")?)].into_py_dict(py)?;
        let code = c"os.getenv('USER') or os.getenv('USERNAME') or 'Unknown'";
        let user: String = py.eval(code, None, Some(&locals))?.extract()?;

        println!("Hello {}, I'm Python {}", user, version);
        list(py).unwrap();
        Ok(())
    })
}

pub fn list<'py>(py: Python<'py>) -> PyResult<()> {
    let x = PyList::empty(py);
    x.append((1, 2, 3, 4, 5, 6))?;
    let y = x.clone();
    println!("{}", y);
    drop(x);
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use pyo3::prelude::*;
    use pyo3::types::*;
    use numpy::{PyArray1, prelude::*};

    #[test]
    pub fn run_gymnasium() -> PyResult<()> {
        Python::attach(|py| {
            let gym = py.import("gymnasium")?;
            let kwargs = PyDict::new(py);
            // kwargs.set_item("render_mode", "human")?;
            let env = gym.call_method("make", ("CartPole-v1",), Some(&kwargs))?;
            let device = Device::default();
            kwargs.clear();
            kwargs.set_item("seed", 12)?;
            env.call_method("reset", (), Some(&kwargs))?;
            for _ in 0..10 {
                let tuple = env.call_method1("step", (0,))?;
                let arr = tuple.get_item(0)?;
                let arr = arr.cast_into::<PyArray1<f32>>()?;
                let ro = arr.readonly();
                let arr = ro.as_array().to_vec();
                let obs: Tensor<2> = Tensor::<1>::from_floats(arr.as_slice(), &device).unsqueeze_dim(0);
                let reward: f32 = tuple.get_item(1)?.extract()?;
                let terminate: bool = tuple.get_item(2)?.extract()?;
                let truncate: bool = tuple.get_item(3)?.extract()?;
                eprintln!("obs: {}, reward: {}, terminated: {}, truncated: {}", obs, reward, terminate, truncate);    
            }
            Ok(())
        })
    }
}