use crate::ranking::bm25::BM25HyperParams;

pub fn get_configuration() -> Result<BM25HyperParams, config::ConfigError> {
    let base_path = std::env::current_dir().expect("Failed to determine the current directory");
    let configuration_directory = base_path.join("configuration");

    let settings = config::Config::builder()
        .add_source(config::File::from(configuration_directory.join("bm25.yml")))
        .build()?;

    settings.try_deserialize::<BM25HyperParams>()
}
