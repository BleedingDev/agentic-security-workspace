//! Largest-first dispatch for the per-module parallel phases.

use rayon::prelude::*;

/// Map `items` in parallel, handing the largest items to workers first, and
/// return the results in the original item order.
///
/// `par_iter` splits the index range recursively, so a large module that sits
/// late in the list can start after most workers have gone idle and become the
/// critical path of the whole phase. Pulling items through `par_bridge` in
/// descending size order is the longest-processing-time-first heuristic: every
/// free worker takes the largest remaining item. Results are re-sorted by
/// original index, so output order and content never depend on scheduling.
pub(super) fn par_map_largest_first<T, R>(
    items: Vec<T>,
    size: impl Fn(&T) -> usize,
    map: impl Fn(T) -> R + Sync + Send,
) -> Vec<R>
where
    T: Send,
    R: Send,
{
    let mut ordered: Vec<(usize, usize, T)> = items
        .into_iter()
        .enumerate()
        .map(|(index, item)| (size(&item), index, item))
        .collect();
    // Largest first; ties keep input order so single-worker dispatch is stable.
    ordered.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut results: Vec<(usize, R)> = ordered
        .into_iter()
        .par_bridge()
        .map(|(_, index, item)| (index, map(item)))
        .collect();
    results.sort_unstable_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, result)| result).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    #[test]
    fn results_keep_input_order_for_any_worker_count() {
        let sizes = vec![3usize, 40, 1, 40, 7, 0, 12, 40, 2];
        let expected: Vec<String> = sizes
            .iter()
            .enumerate()
            .map(|(index, size)| format!("{index}:{size}"))
            .collect();
        for threads in [1, 2, 4] {
            let items: Vec<(usize, usize)> = sizes.iter().copied().enumerate().collect();
            let results = pool(threads).install(|| {
                par_map_largest_first(
                    items,
                    |(_, size)| *size,
                    |(index, size)| format!("{index}:{size}"),
                )
            });
            assert_eq!(results, expected, "{threads} workers");
        }
    }

    #[test]
    fn single_worker_dispatches_largest_first_with_stable_ties() {
        let sizes = vec![3usize, 40, 1, 40, 7, 0, 12, 40, 2];
        let order = Mutex::new(Vec::new());
        let items: Vec<(usize, usize)> = sizes.iter().copied().enumerate().collect();
        pool(1).install(|| {
            par_map_largest_first(
                items,
                |(_, size)| *size,
                |(index, _)| order.lock().unwrap().push(index),
            )
        });
        assert_eq!(order.into_inner().unwrap(), vec![1, 3, 7, 6, 4, 0, 8, 2, 5]);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let results: Vec<u8> =
            par_map_largest_first(Vec::<u8>::new(), |item| *item as usize, |item| item);
        assert!(results.is_empty());
    }
}
