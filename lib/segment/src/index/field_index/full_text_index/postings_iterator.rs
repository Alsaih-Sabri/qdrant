use super::posting_list::PostingList;
use crate::types::PointOffsetType;

pub fn intersect_postings_iterator<'a>(
    mut postings: Vec<&'a PostingList>,
) -> Box<dyn Iterator<Item = PointOffsetType> + 'a> {
    let smallest_posting_idx = postings
        .iter()
        .enumerate()
        .min_by_key(|(_idx, posting)| posting.len())
        .map(|(idx, _posting)| idx)
        .unwrap();
    let smallest_posting = postings.remove(smallest_posting_idx);

    let and_iter = smallest_posting
        .into_iter()
        // .iter()
        .filter(move |doc_id| postings.iter().all(|posting| posting.contains(doc_id)));
    // .copied();

    Box::new(and_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_postings_iterator() {
        let mut p1 = PostingList::default();
        p1.insert(1, 8);
        p1.insert(2, 8);
        p1.insert(3, 8);
        p1.insert(4, 8);
        p1.insert(5, 8);
        let mut p2 = PostingList::default();
        p2.insert(2, 8);
        p2.insert(4, 8);
        p2.insert(5, 8);
        p2.insert(5, 8);
        let mut p3 = PostingList::default();
        p3.insert(1, 8);
        p3.insert(2, 8);
        p3.insert(5, 8);
        p3.insert(6, 8);
        p3.insert(7, 8);

        let postings = vec![&p1, &p2, &p3];
        let merged = intersect_postings_iterator(postings);

        let res = merged.collect::<Vec<_>>();

        assert_eq!(res, vec![2, 5]);
    }
}
