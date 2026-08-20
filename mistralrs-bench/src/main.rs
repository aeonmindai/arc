use candle_core::Device;
use clap::Parser;
use cli_table::{format::Justify, print_stdout, Cell, CellStruct, Style, Table};
use mistralrs_core::{
    get_auto_device_map_params, get_model_dtype, initialize_logging, paged_attn_supported,
    parse_isq_value, Constraint, DefaultSchedulerMethod, DeviceLayerMapMetadata, DeviceMapMetadata,
    DeviceMapSetting, DrySamplingParams, Loader, LoaderBuilder, MemoryGpuConfig, MistralRs,
    MistralRsBuilder, ModelSelected, NormalRequest, PagedAttentionConfig, PagedCacheType, Request,
    RequestMessage, Response, SamplingParams, SchedulerConfig, TokenSource, Usage,
};
use std::fmt::Display;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::channel;
use tracing::{info, warn};

enum TestName {
    Prompt(usize),
    Gen(usize),
}

impl Display for TestName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            TestName::Prompt(n) => format!("pp {n}"),
            TestName::Gen(n) => format!("tg {n}"),
        };
        write!(f, "{name}")
    }
}

struct BenchResult {
    usages: Vec<Usage>,
    concurrency: usize,
    test_name: TestName,
    /// Wall-clock seconds for the whole send->all-Done window, measured by the
    /// client with no engine cooperation. This is the SECOND instrument: the
    /// engine's own `Usage` numbers are only believable when they agree with a
    /// clock that lives outside the engine.
    wall_secs: f32,
    /// Total prompt tokens the engine says it processed across all responses.
    prompt_toks_seen: usize,
}

struct UncertainTokSec {
    mean: f32,
    std_dev: f32,
}

impl Display for UncertainTokSec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3}±{:.3}", self.mean, self.std_dev)
    }
}

async fn run_bench(
    mistralrs: Arc<MistralRs>,
    prompt: RequestMessage,
    n_gen: usize,
    concurrency: usize,
    repetitions: usize,
    test_name: TestName,
) -> anyhow::Result<BenchResult> {
    let sampling_params = SamplingParams {
        temperature: Some(0.1),
        top_k: Some(32),
        top_p: Some(0.1),
        min_p: Some(0.05),
        top_nsigma: None,
        top_n_logprobs: 0,
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.1),
        repetition_penalty: None,
        max_len: Some(n_gen),
        stop_toks: None,
        logits_bias: None,
        n_choices: 1,
        dry_params: Some(DrySamplingParams::default()),
        early_stop_confidence: None,
        reasoning_budget: None,
    };
    let sender = mistralrs.get_sender(None).unwrap();
    let (tx, mut rx) = channel(10_000);

    // 🔴 A fresh request id per send.
    //
    // This used to call `next_request_id()` exactly once and then `.clone()`
    // the same `NormalRequest` for every concurrent copy and every repetition,
    // so `concurrency * repetitions` sequences were in flight sharing ONE id.
    // The id is the engine's handle for a sequence group; issuing duplicates
    // means unrelated sequences alias each other in engine-side bookkeeping.
    // A benchmark must not be the only caller in the system that does this.
    let make_req = |id: usize| {
        Request::Normal(Box::new(NormalRequest {
            id,
            messages: prompt.clone(),
            sampling_params: sampling_params.clone(),
            response: tx.clone(),
            return_logprobs: false,
            is_streaming: false,
            constraint: Constraint::None,
            suffix: None,
            tools: None,
            tool_choice: None,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
            model_id: None,
            truncate_sequence: false,
        }))
    };

    let mut usages = Vec::new();
    let wall_start = Instant::now();

    for _ in 0..repetitions {
        for _ in 0..concurrency {
            let req = make_req(mistralrs.next_request_id());
            if sender.send(req).await.is_err() {
                eprintln!("Receiver disconnected");
            }
        }
        for _ in 0..concurrency {
            match rx.recv().await {
                Some(r) => match r {
                    Response::InternalError(e) => {
                        unreachable!("Got an internal error: {e:?}");
                    }
                    Response::ModelError(e, resp) => {
                        unreachable!("Got a model error: {e:?}, response: {resp:?}");
                    }
                    Response::ValidationError(e) => {
                        unreachable!("Got a validation error: {e:?}");
                    }
                    Response::Done(res) => {
                        usages.push(res.usage);
                    }
                    Response::Chunk(_) => unreachable!(),
                    Response::CompletionModelError(_, _) => unreachable!(),
                    Response::CompletionDone(res) => {
                        usages.push(res.usage);
                    }
                    Response::CompletionChunk(_) => unreachable!(),
                    Response::ImageGeneration(_) => unreachable!(),
                    Response::Speech { .. } => unreachable!(),
                    Response::Raw { .. } => unreachable!(),
                    Response::Embeddings { .. } => unreachable!(),
                },
                None => unreachable!("Expected a Done response, got None",),
            }
        }
    }

    let wall_secs = wall_start.elapsed().as_secs_f32();
    let prompt_toks_seen = usages.iter().map(|u| u.prompt_tokens).sum();

    Ok(BenchResult {
        usages,
        concurrency,
        test_name,
        wall_secs,
        prompt_toks_seen,
    })
}

fn get_tok_s(result: &BenchResult) -> UncertainTokSec {
    let ts_measurements = match result.test_name {
        TestName::Prompt(_) => result
            .usages
            .iter()
            .map(|u| u.avg_prompt_tok_per_sec)
            .collect::<Vec<_>>(),
        TestName::Gen(_) => result
            .usages
            .iter()
            .map(|u| u.avg_compl_tok_per_sec)
            .collect::<Vec<_>>(),
    };
    // Calculate uncertainty
    let mean = ts_measurements.iter().sum::<f32>() / ts_measurements.len() as f32;
    let variance = ts_measurements
        .iter()
        .map(|e| (mean - e).powf(2.))
        .sum::<f32>()
        / ts_measurements.len() as f32;
    let std_dev = variance.sqrt();
    UncertainTokSec { mean, std_dev }
}

fn get_ms_tok(result: &BenchResult) -> UncertainTokSec {
    let ms_tok_measurements = match result.test_name {
        TestName::Prompt(_) => result
            .usages
            .iter()
            .map(|u| 1000. / u.avg_prompt_tok_per_sec)
            .collect::<Vec<_>>(),
        TestName::Gen(_) => result
            .usages
            .iter()
            .map(|u| 1000. / u.avg_compl_tok_per_sec)
            .collect::<Vec<_>>(),
    };
    // Calculate uncertainty
    let mean = ms_tok_measurements.iter().sum::<f32>() / ms_tok_measurements.len() as f32;
    let variance = ms_tok_measurements
        .iter()
        .map(|e| (mean - e).powf(2.))
        .sum::<f32>()
        / ms_tok_measurements.len() as f32;
    let std_dev = variance.sqrt();
    UncertainTokSec { mean, std_dev }
}

/// A zero is not a measurement.
///
/// 🔴 This harness used to print `0.000±0.000` into a results table and exit 0.
/// That is the house fault: a green result that proves no work happened. Every
/// row is now required to carry evidence that the engine actually processed the
/// tokens the row claims to be about, and a row that cannot is a hard stop.
///
/// Exits **2** (environment/instrument failure), never 1, so a caller can tell
/// "the benchmark could not be run" apart from "the benchmark ran and regressed".
fn assert_results_are_measurements(results: &[BenchResult]) {
    let mut failures: Vec<String> = Vec::new();

    for r in results {
        let label = format!("{} @ concurrency {}", r.test_name, r.concurrency);

        if r.usages.is_empty() {
            failures.push(format!("{label}: no responses at all (0 usages collected)"));
            continue;
        }

        for (i, u) in r.usages.iter().enumerate() {
            let where_ = format!("{label}, response {i}");
            match r.test_name {
                TestName::Prompt(n) => {
                    if u.prompt_tokens == 0 {
                        failures.push(format!("{where_}: engine reported 0 prompt tokens"));
                    } else if u.prompt_tokens != n {
                        failures.push(format!(
                            "{where_}: asked for {n} prompt tokens, engine processed {}",
                            u.prompt_tokens
                        ));
                    }
                    if u.total_prompt_time_sec <= 0.0 {
                        failures.push(format!(
                            "{where_}: prompt time is {:.6}s -- the prefill step was never timed",
                            u.total_prompt_time_sec
                        ));
                    }
                    if !u.avg_prompt_tok_per_sec.is_finite() || u.avg_prompt_tok_per_sec <= 0.0 {
                        failures.push(format!(
                            "{where_}: prompt rate is {} tok/s",
                            u.avg_prompt_tok_per_sec
                        ));
                    }
                }
                TestName::Gen(_) => {
                    if u.completion_tokens == 0 {
                        failures.push(format!("{where_}: engine produced 0 completion tokens"));
                    }
                    if !u.avg_compl_tok_per_sec.is_finite() || u.avg_compl_tok_per_sec <= 0.0 {
                        failures.push(format!(
                            "{where_}: completion rate is {} tok/s",
                            u.avg_compl_tok_per_sec
                        ));
                    }
                }
            }
        }
    }

    if !failures.is_empty() {
        eprintln!("\n=== BENCHMARK PRODUCED NO VALID MEASUREMENT ===");
        for f in &failures {
            eprintln!("  - {f}");
        }
        eprintln!(
            "\n{} check(s) failed. A zero-token or zero-rate result is an instrument\n\
             failure, not a score, so no table is printed. Exiting 2.",
            failures.len()
        );
        std::process::exit(2);
    }
}

fn print_usage(model: &str, device: &Device, results: Vec<BenchResult>) {
    let backend = match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        Device::Metal(_) => "Metal",
    };
    let results: Vec<Vec<CellStruct>> = results
        .into_iter()
        .map(|r| {
            vec![
                model.cell(),
                backend.cell(),
                r.test_name.to_string().cell(),
                get_tok_s(&r).cell().justify(Justify::Right),
                get_ms_tok(&r).cell().justify(Justify::Right),
                r.concurrency.cell().justify(Justify::Right),
                (get_tok_s(&r).mean * r.concurrency as f32)
                    .cell()
                    .justify(Justify::Right),
                // Second instrument: a clock outside the engine.
                format!("{:.3}", r.wall_secs).cell().justify(Justify::Right),
                format!("{:.1}", r.prompt_toks_seen as f32 / r.wall_secs)
                    .cell()
                    .justify(Justify::Right),
            ]
        })
        .collect();

    let table = results
        .table()
        .title(vec![
            "model".cell().bold(true),
            // "size".cell().bold(true),
            // "params".cell().bold(true),
            "backend".cell().bold(true),
            // "ngl".cell().bold(true),
            "test".cell().bold(true),
            "t/s".cell().bold(true),
            "ms/t".cell().bold(true),
            "concurrency".cell().bold(true),
            "throughput/s".cell().bold(true),
            "wall s".cell().bold(true),
            "prompt t/s (wall)".cell().bold(true),
        ])
        .bold(true);
    print_stdout(table).expect("print table");
}

async fn warmup_run(mistralrs: Arc<MistralRs>) {
    let sampling_params = SamplingParams {
        max_len: Some(1),
        ..SamplingParams::deterministic()
    };
    let sender = mistralrs.get_sender(None).unwrap();
    let (tx, mut rx) = channel(10_000);

    let req = Request::Normal(Box::new(NormalRequest {
        id: mistralrs.next_request_id(),
        messages: RequestMessage::Completion {
            text: "Hello!".to_string(),
            echo_prompt: false,
            best_of: None,
        },
        sampling_params: sampling_params.clone(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        constraint: Constraint::None,
        suffix: None,
        tools: None,
        tool_choice: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        model_id: None,
        truncate_sequence: false,
    }));

    if sender.send(req.clone()).await.is_err() {
        eprintln!("Receiver disconnected");
    }

    let _ = rx.recv().await;
}

fn parse_cache_type(s: &str) -> Result<PagedCacheType, String> {
    s.parse()
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Model
    #[clap(subcommand)]
    model: ModelSelected,

    /// Integer seed to ensure reproducible random number generation.
    #[arg(short, long)]
    seed: Option<u64>,

    /// Number of prompt tokens to run.
    #[arg(long, short = 'p', default_value_t = 512)]
    n_prompt: usize,

    /// Number of generations tokens to run.
    #[arg(long, short = 'g', default_value_t = 128)]
    n_gen: usize,

    /// Number of concurrent requests to run. Default is 1
    #[clap(short, long, value_parser, value_delimiter = ',')]
    concurrency: Option<Vec<usize>>,

    /// Number of times to repeat each test.
    #[arg(long, short, default_value_t = 5)]
    repetitions: usize,

    /// NOTE: This can be omitted to use automatic device mapping!
    /// Number of device layers to load and run on GPU(s). All others will be on the CPU.
    /// If one GPU is used, then this value should be an integer. Otherwise, it follows the following pattern:
    /// ORD:NUM;... Where ORD is a unique device ordinal and NUM is the number of layers for that device.
    #[arg(short, long, value_parser, value_delimiter = ';')]
    num_device_layers: Option<Vec<String>>,

    /// In-situ quantization to apply.
    #[arg(long = "isq")]
    in_situ_quant: Option<String>,

    /// GPU memory to allocate for KV cache with PagedAttention in MBs.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    #[arg(long = "pa-gpu-mem")]
    paged_attn_gpu_mem: Option<usize>,

    /// Percentage of GPU memory to utilize after allocation of KV cache with PagedAttention, from 0 to 1.
    /// If this is not set and the device is CUDA, it will default to `0.9`.
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    #[arg(long = "pa-gpu-mem-usage")]
    paged_attn_gpu_mem_usage: Option<f32>,

    /// Total context length to allocate the KV cache for (total number of tokens which the KV cache can hold).
    /// PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
    /// The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
    /// This is the default setting, and it defaults to the `max-seq-len` specified in after the model type.
    #[arg(long = "pa-ctxt-len")]
    paged_ctxt_len: Option<usize>,

    /// PagedAttention KV cache type (auto or f8e4m3).
    /// Defaults to `auto`.
    #[arg(long = "pa-cache-type", value_parser = parse_cache_type)]
    cache_type: Option<PagedCacheType>,

    /// Block size (number of tokens per block) for PagedAttention. If this is not set and the device is CUDA, it will default to 32.
    /// PagedAttention is only supported on CUDA and is always automatically activated.
    #[arg(long = "pa-blk-size")]
    paged_attn_block_size: Option<usize>,

    /// Disable PagedAttention on CUDA. Because PagedAttention is already disabled on Metal, this is only applicable on CUDA.
    #[arg(long = "no-paged-attn", default_value_t = false)]
    no_paged_attn: bool,

    /// Enable PagedAttention on Metal. Because PagedAttention is already enabled on CUDA, this is only applicable on Metal.
    #[arg(long = "paged-attn", default_value_t = false)]
    paged_attn: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();
    initialize_logging();

    warn!(
        "mistralrs-bench is deprecated. Please use `mistralrs bench` from mistralrs-cli instead."
    );

    args.concurrency = Some(args.concurrency.unwrap_or(vec![1]));

    let dtype = get_model_dtype(&args.model)?;
    let auto_device_map_params = get_auto_device_map_params(&args.model)?;

    let max_seq_len = auto_device_map_params.max_seq_len();

    let loader: Box<dyn Loader> = LoaderBuilder::new(args.model).build()?;
    let model_name = loader.get_id();

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = if mistralrs_core::distributed::use_nccl() {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };

    if let Some(seed) = args.seed {
        device.set_seed(seed)?;
    }

    let token_source = TokenSource::CacheToken;
    info!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    info!("Sampling method: penalties -> temperature -> topk -> topp -> minp -> multinomial");
    info!("Model kind is: {}", loader.get_kind().to_string());

    // Parse device mapper
    let mapper = if let Some(device_layers) = args.num_device_layers {
        if device_layers.len() == 1 && device_layers[0].parse::<usize>().is_ok() {
            let layers = device_layers[0].parse::<usize>().unwrap();
            DeviceMapSetting::Map(DeviceMapMetadata::from_num_device_layers(vec![
                DeviceLayerMapMetadata { ordinal: 0, layers },
            ]))
        } else {
            let mut mapping = Vec::new();
            for layer in device_layers {
                let split = layer.splitn(2, ':').collect::<Vec<_>>();
                if split.len() < 2 {
                    panic!("Expected layer to be of format ORD:NUM, got {layer}");
                }
                let ord = split[0]
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("Failed to parse {} as integer.", split[0]));
                let num = split[1]
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("Failed to parse {} as integer.", split[1]));
                for DeviceLayerMapMetadata { ordinal, layers: _ } in &mapping {
                    if *ordinal == ord {
                        panic!("Duplicate ordinal {ord}");
                    }
                }
                mapping.push(DeviceLayerMapMetadata {
                    ordinal: ord,
                    layers: num,
                });
            }
            DeviceMapSetting::Map(DeviceMapMetadata::from_num_device_layers(mapping))
        }
    } else {
        DeviceMapSetting::Auto(auto_device_map_params)
    };

    let no_paged_attn = if device.is_cuda() || mistralrs_core::distributed::use_nccl() {
        args.no_paged_attn
    } else if device.is_metal() {
        !args.paged_attn
    } else {
        true
    };

    let cache_config = match (
        args.paged_attn_block_size,
        args.paged_attn_gpu_mem,
        args.paged_attn_gpu_mem_usage,
        args.paged_ctxt_len,
        paged_attn_supported(),
        no_paged_attn,
    ) {
        (block_size, None, None, None, true, false) => Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::ContextSize(max_seq_len),
            args.cache_type.unwrap_or_default(),
        )?),
        (block_size, None, None, Some(ctxt), true, false) => Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::ContextSize(ctxt),
            args.cache_type.unwrap_or_default(),
        )?),
        (block_size, None, Some(f), None, true, false) => Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::Utilization(f),
            args.cache_type.unwrap_or_default(),
        )?),
        (block_size, Some(m), None, None, true, false) => Some(PagedAttentionConfig::new(
            block_size,
            MemoryGpuConfig::MbAmount(m),
            args.cache_type.unwrap_or_default(),
        )?),
        (block_size, Some(_m), Some(f), None, true, false) => {
            info!("Both memory size, and usage were specified, defaulting to the usage value.");
            Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::Utilization(f),
                args.cache_type.unwrap_or_default(),
            )?)
        }
        (block_size, Some(_m), None, Some(ctxt), true, false) => {
            info!("All memory size and ctxt len, defaulting to the context len value.");
            Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::ContextSize(ctxt),
                args.cache_type.unwrap_or_default(),
            )?)
        }
        (block_size, None, Some(f), Some(_ctxt), true, false) => {
            info!("Both ctxt len and usage were specified, defaulting to the usage value.");
            Some(PagedAttentionConfig::new(
                block_size,
                MemoryGpuConfig::Utilization(f),
                args.cache_type.unwrap_or_default(),
            )?)
        }
        (_, _, _, _, _, _) => None,
    };

    let isq = args
        .in_situ_quant
        .as_ref()
        .map(|isq| parse_isq_value(isq, Some(&device)).map_err(|e| anyhow::anyhow!("{e}")))
        .transpose()?;

    let pipeline = loader.load_model_from_hf(
        None,
        token_source,
        &dtype,
        &device,
        false,
        mapper,
        isq,
        cache_config,
    )?;
    info!("Model loaded.");

    let scheduler_config = if cache_config.is_some() {
        // Handle case where we may have device mapping
        if let Some(ref cache_config) = pipeline.lock().await.get_metadata().cache_config {
            SchedulerConfig::PagedAttentionMeta {
                max_num_seqs: *args.concurrency.as_ref().unwrap().iter().max().unwrap(),
                config: cache_config.clone(),
            }
        } else {
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(
                    (*args.concurrency.as_ref().unwrap().iter().max().unwrap())
                        .try_into()
                        .unwrap(),
                ),
            }
        }
    } else {
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(
                (*args.concurrency.as_ref().unwrap().iter().max().unwrap())
                    .try_into()
                    .unwrap(),
            ),
        }
    };
    let mistralrs = MistralRsBuilder::new(pipeline, scheduler_config, false, None)
        .with_no_prefix_cache(true)
        .with_disable_eos_stop(true)
        .build()
        .await;

    info!("Starting warmup run.");
    warmup_run(mistralrs.clone()).await;
    info!("Finished warmup run.");
    info!("Starting benchmarks.");

    for concurrency in args.concurrency.as_ref().unwrap() {
        let mut results = vec![];
        if args.n_gen > 0 {
            let r = run_bench(
                mistralrs.clone(),
                RequestMessage::Completion {
                    text: "Rust".to_string(),
                    echo_prompt: false,
                    best_of: None,
                },
                args.n_gen - 1,
                *concurrency,
                args.repetitions,
                TestName::Gen(args.n_gen),
            )
            .await?;
            results.push(r);
        }

        if args.n_prompt > 0 {
            let tks = (1000..1000 + args.n_prompt as u32).collect();
            let r = run_bench(
                mistralrs.clone(),
                RequestMessage::CompletionTokens(tks),
                1,
                *concurrency,
                args.repetitions,
                TestName::Prompt(args.n_prompt),
            )
            .await?;

            results.push(r);
        }

        assert_results_are_measurements(&results);
        print_usage(&model_name, &device, results);
    }

    Ok(())
}
