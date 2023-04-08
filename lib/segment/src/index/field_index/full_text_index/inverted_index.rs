use std::collections::HashSet;

use patricia_tree::PatriciaMap;
use serde::{Deserialize, Serialize};

use super::posting_list::PostingList;
use super::postings_iterator::intersect_postings_iterator;
use crate::index::field_index::{CardinalityEstimation, PayloadBlockCondition, PrimaryCondition};
use crate::types::{FieldCondition, Match, MatchText, PayloadKeyType, PointOffsetType};

#[derive(Default, Serialize, Deserialize, Clone)]
pub struct Document {
    pub tokens: Vec<usize>,
}

impl Document {
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

pub struct ParsedQuery {
    pub tokens: Vec<Option<usize>>,
}

impl ParsedQuery {
    pub fn check_match(&self, document: &Document) -> bool {
        if self.tokens.contains(&None) {
            return false;
        }
        // Check that all tokens are in document
        self.tokens
            .iter()
            // unwrap crash safety: all tokens exist in the vocabulary if it passes the above check
            .all(|query_token| document.tokens.binary_search(&query_token.unwrap()).is_ok())
    }
}

#[derive(Default)]
pub struct InvertedIndex {
    postings: Vec<Option<PostingList>>,
    pub vocab: PatriciaMap<usize>,
    vocab_count: usize,
    pub point_to_docs: Vec<Option<Document>>,
    pub points_count: usize,
}

impl InvertedIndex {
    pub fn new() -> InvertedIndex {
        Default::default()
    }

    pub fn document_from_tokens(&mut self, tokens: HashSet<String>) -> Document {
        let mut document_tokens = vec![];
        for token in tokens {
            // check if in vocab
            //
            let vocab_idx = match self.vocab.get(&token) {
                Some(&idx) => idx,
                None => {
                    let len = self.vocab_count;
                    self.vocab_count += 1;
                    self.vocab.insert(token, len);
                    len
                }
            };
            document_tokens.push(vocab_idx);
        }

        Document {
            tokens: document_tokens,
        }
    }

    pub fn index_document(&mut self, idx: PointOffsetType, document: Document) {
        self.points_count += 1;
        if self.point_to_docs.len() <= idx as usize {
            self.point_to_docs
                .resize(idx as usize + 1, Default::default());
        }

        for &token_idx in &document.tokens {
            if self.postings.len() <= token_idx {
                self.postings.resize(token_idx + 1, Default::default());
            }
            let posting = self
                .postings
                .get_mut(token_idx)
                .expect("posting must exist even if with None");
            match posting {
                None => *posting = Some(PostingList::new(idx)),
                Some(vec) => vec.insert(idx, self.point_to_docs.len()),
            }
        }
        self.point_to_docs[idx as usize] = Some(document);
    }

    pub fn remove_document(&mut self, idx: PointOffsetType) -> Option<Document> {
        if self.point_to_docs.len() <= idx as usize {
            return None; // Already removed or never actually existed
        }

        let removed_doc = match std::mem::take(&mut self.point_to_docs[idx as usize]) {
            Some(doc) => doc,
            None => return None,
        };

        self.points_count -= 1;

        for removed_token in &removed_doc.tokens {
            // unwrap safety: posting list exists and contains the document id
            let posting = self.postings.get_mut(*removed_token).unwrap();
            if let Some(vec) = posting {
                vec.remove(idx, self.point_to_docs.len());
            }
        }
        Some(removed_doc)
    }

    pub fn filter(&self, query: &ParsedQuery) -> Box<dyn Iterator<Item = PointOffsetType> + '_> {
        let postings_opt: Option<Vec<_>> = query
            .tokens
            .iter()
            .map(|&vocab_idx| match vocab_idx {
                None => None,
                // if a ParsedQuery token was given an index, then it must exist in the vocabulary
                // dictionary. Posting list entry can be None but it exists.
                Some(idx) => self.postings.get(idx).unwrap().as_ref(),
            })
            .collect();
        if postings_opt.is_none() {
            // There are unseen tokens -> no matches
            return Box::new(vec![].into_iter());
        }
        let postings = postings_opt.unwrap();
        if postings.is_empty() {
            // Empty request -> no matches
            return Box::new(vec![].into_iter());
        }
        intersect_postings_iterator(postings)
    }

    pub fn estimate_cardinality(
        &self,
        query: &ParsedQuery,
        condition: &FieldCondition,
    ) -> CardinalityEstimation {
        let postings_opt: Option<Vec<_>> = query
            .tokens
            .iter()
            .map(|&vocab_idx| match vocab_idx {
                None => None,
                // unwrap safety: same as in filter()
                Some(idx) => self.postings.get(idx).unwrap().as_ref(),
            })
            .collect();
        if postings_opt.is_none() {
            // There are unseen tokens -> no matches
            return CardinalityEstimation {
                primary_clauses: vec![PrimaryCondition::Condition(condition.clone())],
                min: 0,
                exp: 0,
                max: 0,
            };
        }
        let postings = postings_opt.unwrap();
        if postings.is_empty() {
            // Empty request -> no matches
            return CardinalityEstimation {
                primary_clauses: vec![PrimaryCondition::Condition(condition.clone())],
                min: 0,
                exp: 0,
                max: 0,
            };
        }
        // Smallest posting is the largest possible cardinality
        let smallest_posting = postings.iter().map(|posting| posting.len()).min().unwrap();

        return if postings.len() == 1 {
            CardinalityEstimation {
                primary_clauses: vec![PrimaryCondition::Condition(condition.clone())],
                min: smallest_posting,
                exp: smallest_posting,
                max: smallest_posting,
            }
        } else {
            let expected_frac: f64 = postings
                .iter()
                .map(|posting| posting.len() as f64 / self.points_count as f64)
                .product();
            let exp = (expected_frac * self.points_count as f64) as usize;
            CardinalityEstimation {
                primary_clauses: vec![PrimaryCondition::Condition(condition.clone())],
                min: 0, // ToDo: make better estimation
                exp,
                max: smallest_posting,
            }
        };
    }

    pub fn payload_blocks(
        &self,
        threshold: usize,
        key: PayloadKeyType,
    ) -> Box<dyn Iterator<Item = PayloadBlockCondition> + '_> {
        // It might be very hard to predict possible combinations of conditions,
        // so we only build it for individual tokens
        Box::new(
            self.vocab
                .iter()
                .filter(|(_token, &posting_idx)| self.postings[posting_idx].is_some())
                .filter(move |(_token, &posting_idx)| {
                    // unwrap crash safety: all tokens that passes the first filter should have postings
                    self.postings[posting_idx].as_ref().unwrap().len() >= threshold
                })
                .map(|(token, &posting_idx)| {
                    (
                        std::string::String::from_utf8(token).expect("token is not valid utf-8"),
                        // same as the above case
                        self.postings[posting_idx].as_ref().unwrap(),
                    )
                })
                .map(move |(token, posting)| PayloadBlockCondition {
                    condition: FieldCondition {
                        key: key.clone(),
                        r#match: Some(Match::Text(MatchText { text: token })),
                        range: None,
                        geo_bounding_box: None,
                        geo_radius: None,
                        values_count: None,
                    },
                    cardinality: posting.len(),
                }),
        )
    }
}
