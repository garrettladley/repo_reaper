#[derive(
    Hash, Eq, PartialEq, Ord, PartialOrd, Debug, Clone, serde::Serialize, serde::Deserialize,
)]
pub struct Term(pub String);
