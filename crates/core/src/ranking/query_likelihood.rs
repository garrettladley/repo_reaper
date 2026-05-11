use crate::{
    index::{DocId, InvertedIndex, Term},
    query::AnalyzedQuery,
    ranking::{Score, Scored},
};

#[derive(Debug, Clone, PartialEq)]
pub struct QueryLikelihoodParams {
    pub smoothing: QueryLikelihoodSmoothing,
}

impl QueryLikelihoodParams {
    pub fn dirichlet_defaults() -> Self {
        Self {
            smoothing: QueryLikelihoodSmoothing::Dirichlet { mu: 1_500.0 },
        }
    }

    pub fn jelinek_mercer_defaults() -> Self {
        Self {
            smoothing: QueryLikelihoodSmoothing::JelinekMercer { lambda: 0.2 },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QueryLikelihoodSmoothing {
    JelinekMercer { lambda: f64 },
    Dirichlet { mu: f64 },
}

pub struct QueryLikelihood {
    pub params: QueryLikelihoodParams,
}

impl QueryLikelihood {
    pub fn score(&self, index: &InvertedIndex, query: &AnalyzedQuery) -> Scored {
        let scores = index
            .documents()
            .map(|metadata| Score {
                doc_path: metadata.path.clone(),
                score: self.score_doc(index, query, metadata.id),
            })
            .collect();

        Scored(scores)
    }

    fn score_doc(&self, index: &InvertedIndex, query: &AnalyzedQuery, doc_id: DocId) -> f64 {
        query
            .terms()
            .map(|(term, query_term)| {
                query_term.weight * self.smoothed_term_probability(index, doc_id, term).ln()
            })
            .sum()
    }

    pub fn smoothed_term_probability(
        &self,
        index: &InvertedIndex,
        doc_id: DocId,
        term: &Term,
    ) -> f64 {
        let tf = index
            .get_postings(term)
            .and_then(|documents| documents.get(&doc_id))
            .map_or(0.0, |term_doc| term_doc.term_freq as f64);
        let doc_len = index
            .document(doc_id)
            .map_or(0.0, |metadata| metadata.token_length as f64);
        let collection_probability = collection_probability(index, term);

        match self.params.smoothing {
            QueryLikelihoodSmoothing::JelinekMercer { lambda } => {
                let document_probability = if doc_len == 0.0 { 0.0 } else { tf / doc_len };
                (1.0 - lambda) * document_probability + lambda * collection_probability
            }
            QueryLikelihoodSmoothing::Dirichlet { mu } => {
                (tf + mu * collection_probability) / (doc_len + mu)
            }
        }
        .max(f64::MIN_POSITIVE)
    }
}

fn collection_probability(index: &InvertedIndex, term: &Term) -> f64 {
    let total_tokens = index.total_token_count() as f64;
    if total_tokens == 0.0 {
        return f64::MIN_POSITIVE;
    }

    let collection_frequency = index.collection_frequency(term) as f64;
    if collection_frequency > 0.0 {
        return collection_frequency / total_tokens;
    }

    1.0 / (total_tokens + index.vocabulary_size() as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use super::{QueryLikelihood, QueryLikelihoodParams, QueryLikelihoodSmoothing};
    use crate::{
        index::{InvertedIndex, Term},
        query::AnalyzedQuery,
        ranking::RankingAlgo,
    };

    fn index() -> InvertedIndex {
        InvertedIndex::from_documents(&[
            ("rare.rs", &[("rare", 3), ("common", 1)]),
            ("common.rs", &[("common", 6)]),
            ("missing.rs", &[("other", 4)]),
        ])
    }

    fn query(term: &str) -> AnalyzedQuery {
        AnalyzedQuery::from_frequencies(term, HashMap::from([(Term(term.to_string()), 1)]))
    }

    #[test]
    fn missing_terms_are_smoothed() {
        let index = index();
        let doc_id = index.doc_id(PathBuf::from("rare.rs").as_path()).unwrap();
        let scorer = QueryLikelihood {
            params: QueryLikelihoodParams::dirichlet_defaults(),
        };

        let probability = scorer.smoothed_term_probability(
            &index,
            doc_id,
            &Term("not_in_collection".to_string()),
        );

        assert!(probability > 0.0);
    }

    #[test]
    fn rare_term_ranks_matching_document_first() {
        let ranking = RankingAlgo::QueryLikelihood(QueryLikelihoodParams::dirichlet_defaults())
            .rank(&index(), &query("rare"), 3)
            .unwrap()
            .0;

        assert_eq!(ranking[0].doc_path, PathBuf::from("rare.rs"));
    }

    #[test]
    fn common_term_prefers_document_with_more_common_evidence() {
        let ranking = RankingAlgo::QueryLikelihood(QueryLikelihoodParams::dirichlet_defaults())
            .rank(&index(), &query("common"), 3)
            .unwrap()
            .0;

        assert_eq!(ranking[0].doc_path, PathBuf::from("common.rs"));
    }

    #[test]
    fn jelinek_mercer_lambda_controls_background_smoothing() {
        let index = index();
        let doc_id = index.doc_id(PathBuf::from("missing.rs").as_path()).unwrap();
        let term = Term("rare".to_string());
        let low_background = QueryLikelihood {
            params: QueryLikelihoodParams {
                smoothing: QueryLikelihoodSmoothing::JelinekMercer { lambda: 0.1 },
            },
        };
        let high_background = QueryLikelihood {
            params: QueryLikelihoodParams {
                smoothing: QueryLikelihoodSmoothing::JelinekMercer { lambda: 0.9 },
            },
        };

        let low = low_background.smoothed_term_probability(&index, doc_id, &term);
        let high = high_background.smoothed_term_probability(&index, doc_id, &term);

        assert!(high > low);
    }

    #[test]
    fn dirichlet_mu_controls_background_smoothing() {
        let index = index();
        let doc_id = index.doc_id(PathBuf::from("missing.rs").as_path()).unwrap();
        let term = Term("rare".to_string());
        let low_mu = QueryLikelihood {
            params: QueryLikelihoodParams {
                smoothing: QueryLikelihoodSmoothing::Dirichlet { mu: 100.0 },
            },
        };
        let high_mu = QueryLikelihood {
            params: QueryLikelihoodParams {
                smoothing: QueryLikelihoodSmoothing::Dirichlet { mu: 10_000.0 },
            },
        };

        let low = low_mu.smoothed_term_probability(&index, doc_id, &term);
        let high = high_mu.smoothed_term_probability(&index, doc_id, &term);

        assert!(high > low);
    }
}
