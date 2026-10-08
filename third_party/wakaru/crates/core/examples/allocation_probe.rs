//! Count allocator requests around a rule or the pipeline, without timing it.
//! Usage: allocation_probe <rule|pipeline|unpack> <input.js> <output>
//! Rule/pipeline mode excludes parsing, resolution, fixing, printing, reporting.
//! Unpack mode includes the complete core unpack operation with one worker;
//! output serialization and disk writes are outside the counter.
//! This executable uses System; it does not change Wakaru's CLI allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

use swc_core::common::{sync::Lrc, FileName, Mark, SourceMap, GLOBALS};
use swc_core::ecma::codegen::{text_writer::JsWriter, Config, Emitter};
use swc_core::ecma::parser::{lexer::Lexer, EsSyntax, Parser, StringInput, Syntax};
use swc_core::ecma::transforms::base::{fixer::fixer, resolver};
use swc_core::ecma::visit::VisitMutWith;
use wakaru_core::rules::UnObjectSpread;
use wakaru_core::{apply_rules, RulePipelineOptions};

static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static REALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

fn record(ptr: *mut u8, size: usize, counter: &AtomicUsize) {
    if !ptr.is_null() && ENABLED.load(Relaxed) {
        counter.fetch_add(1, Relaxed);
        BYTES.fetch_add(size, Relaxed);
    }
}

// No allocation, formatting, or locks in the accounting path. The probe runs
// one operation at a time; these are process-wide requested bytes, not RSS or
// peak live bytes. Each successful realloc counts its entire requested size.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        record(ptr, layout.size(), &ALLOCS);
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        record(ptr, layout.size(), &ALLOCS);
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, layout, size) };
        record(ptr, size, &REALLOCS);
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Debug, PartialEq, Eq)]
struct Counts {
    allocations: usize,
    reallocations: usize,
    requested_bytes: usize,
}

fn measure(run: impl FnOnce()) -> Counts {
    assert!(!ENABLED.load(Relaxed), "nested measurement");
    ALLOCS.store(0, Relaxed);
    REALLOCS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    struct Disable;
    impl Drop for Disable {
        fn drop(&mut self) {
            ENABLED.store(false, Relaxed);
        }
    }
    let guard = Disable;
    ENABLED.store(true, Relaxed);
    run();
    drop(guard);
    Counts {
        allocations: ALLOCS.load(Relaxed),
        reallocations: REALLOCS.load(Relaxed),
        requested_bytes: BYTES.load(Relaxed),
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "expected <rule|pipeline|unpack> <input.js> <output>"
    );
    assert!(matches!(args[1].as_str(), "rule" | "pipeline" | "unpack"));
    let source = std::fs::read_to_string(&args[2]).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let mut samples = Vec::new();
    let mut expected_output = None;
    // First run warms atom interning and lazy runtime state; retain every
    // subsequent sample so drift in allocation counts remains visible.
    for iteration in 0..6 {
        let (counts, output) = if args[1] == "unpack" {
            let mut result = None;
            let counts = pool.install(|| {
                measure(|| {
                    result = Some(
                        wakaru_core::driver::test_support::unpack(
                            &source,
                            wakaru_core::DecompileOptions::default(),
                        )
                        .expect("bundle must unpack"),
                    );
                })
            });
            let result = result.unwrap();
            (counts, serde_json::to_vec(&result.modules).unwrap())
        } else {
            GLOBALS.set(&Default::default(), || {
                let cm: Lrc<SourceMap> = Default::default();
                let file =
                    cm.new_source_file(FileName::Custom(args[2].clone()).into(), source.clone());
                let lexer = Lexer::new(
                    Syntax::Es(EsSyntax {
                        jsx: true,
                        ..Default::default()
                    }),
                    Default::default(),
                    StringInput::from(&*file),
                    None,
                );
                let mut parser = Parser::new_from(lexer);
                let mut module = parser.parse_module().expect("input must parse");
                assert!(parser.take_errors().is_empty(), "recoverable parse errors");
                let unresolved = Mark::new();
                module.visit_mut_with(&mut resolver(unresolved, Mark::new(), false));
                let counts = measure(|| {
                    if args[1] == "rule" {
                        module.visit_mut_with(&mut UnObjectSpread::new_with_mark(unresolved));
                    } else {
                        apply_rules(&mut module, unresolved, RulePipelineOptions::default());
                    }
                });
                module.visit_mut_with(&mut fixer(None));
                let mut output = Vec::new();
                Emitter {
                    cfg: Config::default().with_minify(false),
                    cm: cm.clone(),
                    comments: None,
                    wr: JsWriter::new(cm, "\n", &mut output, None),
                }
                .emit_module(&module)
                .unwrap();
                (counts, output)
            })
        };
        if let Some(expected) = &expected_output {
            assert_eq!(&output, expected, "output changed between probe runs");
        } else {
            expected_output = Some(output);
        }
        if iteration > 0 {
            samples.push(serde_json::json!({
                "allocations": counts.allocations,
                "reallocations": counts.reallocations,
                "requested_bytes": counts.requested_bytes,
            }));
        }
    }
    std::fs::write(&args[3], expected_output.unwrap()).unwrap();
    println!(
        "{}",
        serde_json::json!({"mode": args[1], "samples": samples})
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_successful_requests_and_excludes_disabled_allocations() {
        let counts = measure(|| unsafe {
            let initial = Layout::from_size_align(100, 8).unwrap();
            let ptr = ALLOCATOR.alloc(initial);
            assert!(!ptr.is_null());
            let ptr = ALLOCATOR.realloc(ptr, initial, 200);
            assert!(!ptr.is_null());
            ALLOCATOR.dealloc(ptr, Layout::from_size_align(200, 8).unwrap());
            let zeroed = Layout::from_size_align(16, 8).unwrap();
            let ptr = ALLOCATOR.alloc_zeroed(zeroed);
            assert!(!ptr.is_null());
            ALLOCATOR.dealloc(ptr, zeroed);
        });
        assert_eq!(
            counts,
            Counts {
                allocations: 2,
                reallocations: 1,
                requested_bytes: 316
            }
        );
        let outside = std::hint::black_box(vec![1u8; 128]);
        drop(outside);
        assert_eq!(
            measure(|| {}),
            Counts {
                allocations: 0,
                reallocations: 0,
                requested_bytes: 0
            }
        );
    }
}
