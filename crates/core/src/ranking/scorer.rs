use std::{collections::HashMap, path::PathBuf, str::FromStr};

use dashmap::DashMap;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    index::{DocId, InvertedIndex, Term, TermDocument},
    query::{AnalyzedQuery, QueryTerm},
    ranking::{BM25, BM25HyperParams, CosineSimilarity, TFIDF, get_configuration},
};

#[derive(Debug)]
pub struct Scored(pub Vec<Score>);

impl std::fmt::Display for Scored {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();

        self.0.iter().for_each(|score| {
            output.push_str(&format!("{}\n", score));
        });

        write!(f, "{}", output)
    }
}

#[derive(Debug)]
pub struct Score {
    pub doc_path: PathBuf,
    pub score: f64,
}

impl std::fmt::Display for Score {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Path: {} Score: {}", self.doc_path.display(), self.score)
    }
}

#[derive(Debug, Clone)]
pub enum RankingAlgo {
    CosineSimilarity,
    BM25(BM25HyperParams),
    TFIDF,
}

pub trait RankingAlgorithm {
    fn score(&self, inverted_index: &InvertedIndex, query: &AnalyzedQuery) -> Scored;
}

pub trait Scorer: Send + Sync {
    fn score(
        &self,
        inverted_index: &InvertedIndex,
        query: &AnalyzedQuery,
        term: &Term,
        query_term: QueryTerm,
        documents: &HashMap<DocId, TermDocument>,
        scores: &DashMap<DocId, f64>,
    );
}

impl RankingAlgorithm for RankingAlgo {
    fn score(&self, inverted_index: &InvertedIndex, query: &AnalyzedQuery) -> Scored {
        match self {
            RankingAlgo::CosineSimilarity => score_with(CosineSimilarity, inverted_index, query),
            RankingAlgo::BM25(hyper_params) => score_with(
                BM25 {
                    hyper_params: hyper_params.clone(),
                },
                inverted_index,
                query,
            ),
            RankingAlgo::TFIDF => score_with(TFIDF, inverted_index, query),
        }
    }
}

fn score_with<S>(scorer: S, inverted_index: &InvertedIndex, query: &AnalyzedQuery) -> Scored
where
    S: Scorer,
{
    let scores = DashMap::new();

    query.terms().par_bridge().for_each(|(term, query_term)| {
        if let Some(documents) = inverted_index.get_postings(term) {
            scorer.score(inverted_index, query, term, *query_term, documents, &scores)
        }
    });

    Scored(
        scores
            .iter()
            .par_bridge()
            .filter_map(|score| {
                inverted_index.document(*score.key()).map(|metadata| Score {
                    doc_path: metadata.path.clone(),
                    score: *score.value(),
                })
            })
            .collect(),
    )
}

impl RankingAlgo {
    pub fn rank(
        &self,
        inverted_index: &InvertedIndex,
        query: &AnalyzedQuery,
        top_n: usize,
    ) -> Option<Scored> {
        if query.is_empty() {
            return None;
        }

        let mut ranking = self.score(inverted_index, query).0;

        ranking.sort_by(|a, b| b.score.total_cmp(&a.score));
        ranking.truncate(top_n);

        if ranking.is_empty() {
            None
        } else {
            Some(Scored(ranking))
        }
    }
}

impl FromStr for RankingAlgo {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cosim" => Ok(RankingAlgo::CosineSimilarity),
            "bm25" => {
                let hyper_params =
                    get_configuration().map_err(|e| format!("failed to load BM25 config: {e}"))?;

                Ok(RankingAlgo::BM25(hyper_params))
            }
            "tfidf" => Ok(RankingAlgo::TFIDF),
            _ => Err(format!("{} is not a valid ranking algorithm", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use super::{BM25HyperParams, RankingAlgo};
    use crate::{
        index::{InvertedIndex, Term},
        query::AnalyzedQuery,
    };

    fn index() -> InvertedIndex {
        InvertedIndex::from_documents(&[
            ("rust.rs", &[("rust", 5), ("code", 1)]),
            ("code.rs", &[("rust", 1), ("code", 5)]),
        ])
    }

    fn query_with_weights(rust: f64, code: f64) -> AnalyzedQuery {
        AnalyzedQuery::from_weights(
            "weighted query",
            HashMap::from([
                (Term("rust".to_string()), rust),
                (Term("code".to_string()), code),
            ]),
        )
    }

    fn bm25() -> RankingAlgo {
        RankingAlgo::BM25(BM25HyperParams { k1: 1.2, b: 0.75 })
    }

    #[test]
    fn bm25_query_weights_change_ranking_order() {
        let index = index();
        let rust_weighted = query_with_weights(4.0, 1.0);
        let code_weighted = query_with_weights(1.0, 4.0);

        let rust_top = bm25().rank(&index, &rust_weighted, 1).unwrap().0;
        let code_top = bm25().rank(&index, &code_weighted, 1).unwrap().0;

        assert_eq!(rust_top[0].doc_path, PathBuf::from("rust.rs"));
        assert_eq!(code_top[0].doc_path, PathBuf::from("code.rs"));
    }

    #[test]
    fn tfidf_query_weights_change_ranking_order() {
        let index = index();
        let rust_weighted = query_with_weights(4.0, 1.0);
        let code_weighted = query_with_weights(1.0, 4.0);

        let rust_top = RankingAlgo::TFIDF
            .rank(&index, &rust_weighted, 1)
            .unwrap()
            .0;
        let code_top = RankingAlgo::TFIDF
            .rank(&index, &code_weighted, 1)
            .unwrap()
            .0;

        assert_eq!(rust_top[0].doc_path, PathBuf::from("rust.rs"));
        assert_eq!(code_top[0].doc_path, PathBuf::from("code.rs"));
    }
}
