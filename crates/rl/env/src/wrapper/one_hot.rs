//! Struct for converting Tabular environments into one-hot encoded observation environment
//! 
use crate::{DiscreteEnvironmentConfiguration, Environment, EnvironmentConfiguration, TabularEnvironmentConfiguration};

/// Environment for one-hot encoded observation
pub struct OneHotEncodedEnvironment<E>
where E: Environment<EnvConfig: TabularEnvironmentConfiguration>
{
    env: E
}