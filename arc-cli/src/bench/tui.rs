//! Ratatui frontend for `arc bench --suite agentperf`.
//!
//! Renders a live dashboard while the scheduler runs:
//!
//! ```text
//! ┌─ arc bench: agentperf ───────────────────────────────────┐
//! │ Model: <model>                                           │
//! │ SLO tier: 2  (P25 >= 60 tok/s, P95 TTFT <= 2.0s)         │
//! │ Vendor: <name>                                           │
//! ├──────────────────────────────────────────────────────────┤
//! │ Phase: ramp #3        Concurrent users: 4                │
//! │ Elapsed: 02:14        Phase remaining: ~00:18            │
//! ├──────────────────────────────────────────────────────────┤
//! │  P25 output speed:  108.4 tok/s  ████████░░  [SLO pass]  │
//! │  P95 TTFT:           0.81 s      ███████░░░  [SLO pass]  │
//! │  System throughput:  421.0 tok/s                         │
//! │  Active requests:   4  In-flight tokens: 80              │
//! ├──────────────────────────────────────────────────────────┤
//! │ Phase history                                            │
//! │  #1 K=1   ramp   P25=120.3 P95=0.55  PASS                │
//! │  #2 K=2   ramp   P25=115.9 P95=0.62  PASS                │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! The TUI is driven by a `broadcast::Receiver<Event>` from the scheduler.
//! `q` quits gracefully (we still join the scheduler before exiting so the
//! final JSON / markdown are written).

use crate::bench::scheduler::{Event, PhaseRecord, ScheduleReport};
use crate::bench::slo::SloTier;
use anyhow::{Context, Result};
use ratatui::crossterm::event::{self, Event as CtEvent, KeyCode, KeyEventKind};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{self, Stdout};
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

/// Snapshot of state the TUI renders. Updated from scheduler events.
#[derive(Debug, Default, Clone)]
struct TuiState {
    model: String,
    vendor: String,
    tier: Option<SloTier>,
    max_users_cap: u32,
    bench_start: Option<Instant>,

    current_phase_index: u32,
    current_phase_kind: String,
    current_phase_users: u32,
    current_phase_start: Option<Instant>,
    current_phase_warmup: Duration,
    current_phase_steady_state: Duration,
    in_warmup: bool,
    elapsed_in_phase: Duration,
    running_p25_speed: f64,
    running_p95_ttft: f64,
    running_system_throughput: f64,
    active_requests: u32,
    steady_state_samples: u32,

    history: Vec<PhaseRecord>,
    final_report: Option<ScheduleReport>,
    finished: bool,
}

impl TuiState {
    fn apply(&mut self, ev: Event) {
        match ev {
            Event::Started { tier, total_max_users } => {
                self.tier = Some(tier);
                self.max_users_cap = total_max_users;
                self.bench_start = Some(Instant::now());
            }
            Event::PhaseStarted {
                phase_index,
                concurrent_users,
                kind,
                warmup,
                steady_state,
            } => {
                self.current_phase_index = phase_index;
                self.current_phase_kind = kind;
                self.current_phase_users = concurrent_users;
                self.current_phase_start = Some(Instant::now());
                self.current_phase_warmup = warmup;
                self.current_phase_steady_state = steady_state;
                self.in_warmup = true;
                self.elapsed_in_phase = Duration::ZERO;
                self.running_p25_speed = 0.0;
                self.running_p95_ttft = 0.0;
                self.running_system_throughput = 0.0;
                self.active_requests = 0;
                self.steady_state_samples = 0;
            }
            Event::PhaseProgress {
                phase_index,
                elapsed_in_phase,
                in_warmup,
                steady_state_samples_so_far,
                running_p25_speed,
                running_p95_ttft,
                running_system_throughput_tok_per_s,
                active_requests,
            } => {
                if phase_index == self.current_phase_index {
                    self.elapsed_in_phase = elapsed_in_phase;
                    self.in_warmup = in_warmup;
                    self.steady_state_samples = steady_state_samples_so_far;
                    self.running_p25_speed = running_p25_speed;
                    self.running_p95_ttft = running_p95_ttft;
                    self.running_system_throughput = running_system_throughput_tok_per_s;
                    self.active_requests = active_requests;
                }
            }
            Event::PhaseFinished { record } => {
                self.history.push(record);
            }
            Event::Finished { report } => {
                self.final_report = Some(report);
                self.finished = true;
            }
        }
    }
}

/// RAII guard that restores the terminal even on panic. Borrowed pattern
/// from ratatui's examples: anything that mutates terminal mode goes in
/// the constructor; the destructor unwinds it.
pub struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    pub fn new() -> Result<Self> {
        enable_raw_mode().context("failed to enable raw mode")?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen).context("failed to enter alt screen")?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend).context("failed to construct terminal")?;
        Ok(Self { terminal })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

/// Format a Duration as MM:SS.
fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}", s / 60, s % 60)
}

/// Run the TUI event loop. Returns when the scheduler emits `Finished` or
/// the user presses `q`.
pub async fn run_tui(
    model: &str,
    vendor_name: &str,
    mut rx: broadcast::Receiver<Event>,
) -> Result<()> {
    let mut state = TuiState::default();
    state.model = model.to_string();
    state.vendor = vendor_name.to_string();

    let mut guard = TerminalGuard::new()?;

    let tick = Duration::from_millis(150);
    let mut last_draw = Instant::now() - tick; // force initial draw

    loop {
        // Drain any pending scheduler events without blocking.
        loop {
            match rx.try_recv() {
                Ok(ev) => state.apply(ev),
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
                Err(broadcast::error::TryRecvError::Closed) => break,
            }
        }

        // Re-draw if enough time has elapsed.
        if last_draw.elapsed() >= tick {
            guard.terminal.draw(|f| draw(f, &state))?;
            last_draw = Instant::now();
        }

        // Poll for keyboard input with a short timeout.
        if event::poll(Duration::from_millis(20))? {
            if let CtEvent::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press
                    && (k.code == KeyCode::Char('q') || k.code == KeyCode::Esc)
                {
                    break;
                }
            }
        }

        // Exit on terminal once we've shown the final report for a beat.
        if state.finished {
            // One last draw with the final state.
            guard.terminal.draw(|f| draw(f, &state))?;
            // Give the user ~750 ms to perceive the final state.
            tokio::time::sleep(Duration::from_millis(750)).await;
            break;
        }
    }

    Ok(())
}

/// One-shot draw — pure layout + render. Kept reasonably small so it
/// re-runs every 150 ms without sweating.
fn draw(f: &mut ratatui::Frame<'_>, s: &TuiState) {
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " arc bench: agentperf ",
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::Cyan),
        ));
    let area = f.area();
    f.render_widget(outer.clone(), area);
    let inner = outer.inner(area);

    // Vertical split: header, phase, gauges, history.
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4), // header
            Constraint::Length(3), // phase
            Constraint::Length(7), // gauges
            Constraint::Min(3),    // history
        ])
        .split(inner);

    draw_header(f, chunks[0], s);
    draw_phase(f, chunks[1], s);
    draw_gauges(f, chunks[2], s);
    draw_history(f, chunks[3], s);
}

fn draw_header(f: &mut ratatui::Frame<'_>, area: Rect, s: &TuiState) {
    let tier_line = match s.tier {
        Some(t) => format!(
            "SLO tier: {}  (P25 >= {:.0} tok/s, P95 TTFT <= {:.2} s)",
            t.tier, t.min_p25_output_speed_tok_per_s, t.max_p95_ttft_seconds
        ),
        None => "SLO tier: (not yet set)".to_string(),
    };
    let lines = vec![
        Line::from(vec![
            Span::styled("Model:  ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(&s.model),
        ]),
        Line::from(tier_line),
        Line::from(vec![
            Span::styled("Vendor: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(&s.vendor),
            Span::raw(format!("    Max users cap: {}", s.max_users_cap)),
        ]),
    ];
    f.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_phase(f: &mut ratatui::Frame<'_>, area: Rect, s: &TuiState) {
    let bench_elapsed = s
        .bench_start
        .map(|t| t.elapsed())
        .unwrap_or(Duration::ZERO);
    let phase_total = s.current_phase_warmup + s.current_phase_steady_state;
    let phase_remaining = phase_total.saturating_sub(s.elapsed_in_phase);

    let phase_label = if s.current_phase_index == 0 {
        "warming up...".to_string()
    } else {
        let stage = if s.in_warmup { "warmup" } else { "steady-state" };
        format!(
            "Phase: {} #{}  ({stage})    Concurrent users: {}",
            s.current_phase_kind, s.current_phase_index, s.current_phase_users
        )
    };
    let timing = format!(
        "Elapsed: {}    Phase remaining: ~{}",
        fmt_dur(bench_elapsed),
        fmt_dur(phase_remaining)
    );
    let lines = vec![Line::from(phase_label), Line::from(timing)];
    f.render_widget(Paragraph::new(lines), area);
}

fn pass_label(pass: bool) -> Span<'static> {
    if pass {
        Span::styled(
            "  [SLO pass]",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(
            "  [SLO fail]",
            Style::default()
                .fg(Color::Red)
                .add_modifier(Modifier::BOLD),
        )
    }
}

/// Render a single labelled "metric + gauge" line.
fn render_metric_row(
    f: &mut ratatui::Frame<'_>,
    area: Rect,
    label: &str,
    value_str: String,
    ratio_for_gauge: f64,
    pass: bool,
) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(24),
            Constraint::Min(10),
            Constraint::Length(14),
        ])
        .split(area);
    f.render_widget(
        Paragraph::new(format!("{label}: {value_str}")),
        chunks[0].union(chunks[1]).intersection(chunks[0]),
    );
    // overwrite middle with gauge
    let gauge = Gauge::default()
        .gauge_style(
            Style::default().fg(if pass { Color::Green } else { Color::Red }),
        )
        .ratio(ratio_for_gauge.clamp(0.0, 1.0));
    f.render_widget(gauge, chunks[1]);
    f.render_widget(Paragraph::new(Line::from(pass_label(pass))), chunks[2]);
}

fn draw_gauges(f: &mut ratatui::Frame<'_>, area: Rect, s: &TuiState) {
    let tier = match s.tier {
        Some(t) => t,
        None => {
            f.render_widget(Paragraph::new("(waiting for first phase…)"), area);
            return;
        }
    };
    // 4 lines.
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);

    // P25 output speed — ratio relative to tier floor (clamp at 2x = 1.0).
    let speed_pass = s.running_p25_speed >= tier.min_p25_output_speed_tok_per_s;
    let speed_ratio = (s.running_p25_speed / (tier.min_p25_output_speed_tok_per_s * 2.0))
        .clamp(0.0, 1.0);
    render_metric_row(
        f,
        rows[0],
        "P25 output speed",
        format!("{:>6.1} tok/s", s.running_p25_speed),
        speed_ratio,
        speed_pass,
    );

    // P95 TTFT — inverted ratio (lower TTFT = fuller bar).
    let ttft_pass = s.running_p95_ttft <= tier.max_p95_ttft_seconds;
    let ttft_ratio = if s.running_p95_ttft <= 0.0 {
        1.0
    } else {
        (tier.max_p95_ttft_seconds / (s.running_p95_ttft * 2.0)).clamp(0.0, 1.0)
    };
    render_metric_row(
        f,
        rows[1],
        "P95 TTFT",
        format!("{:>6.2} s    ", s.running_p95_ttft),
        ttft_ratio,
        ttft_pass,
    );

    let sys_text = format!(
        "System throughput: {:>6.1} tok/s",
        s.running_system_throughput
    );
    f.render_widget(Paragraph::new(sys_text), rows[2]);
    let inflight = format!(
        "Active requests: {}    Steady-state samples: {}",
        s.active_requests, s.steady_state_samples
    );
    f.render_widget(Paragraph::new(inflight), rows[3]);
}

fn draw_history(f: &mut ratatui::Frame<'_>, area: Rect, s: &TuiState) {
    let title = if s.finished {
        " Phase history  (FINISHED) "
    } else {
        " Phase history "
    };
    let block = Block::default().borders(Borders::TOP).title(Span::styled(
        title,
        Style::default().add_modifier(Modifier::BOLD),
    ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut items: Vec<ListItem> = s
        .history
        .iter()
        .map(|p| {
            let pass = p.evaluation.overall_pass;
            let badge = if pass {
                Span::styled(" PASS ", Style::default().fg(Color::Green))
            } else {
                Span::styled(" FAIL ", Style::default().fg(Color::Red))
            };
            ListItem::new(Line::from(vec![
                Span::raw(format!(
                    "#{:>2}  K={:>3}  {:<7}  P25={:>5.1}  P95={:>4.2}  N={:>3}",
                    p.phase_index,
                    p.concurrent_users,
                    p.kind,
                    p.evaluation.p25_output_speed_tok_per_s,
                    p.evaluation.p95_ttft_seconds,
                    p.steady_state_samples
                )),
                badge,
            ]))
        })
        .collect();

    if let Some(report) = &s.final_report {
        let summary = match report.saturation_users {
            Some(k) => format!(
                "  >> Saturation: K={k} passes tier {}; max explored K={}",
                report.tier.tier, report.max_users_explored
            ),
            None => "  >> Saturation: even K=1 failed".to_string(),
        };
        items.push(ListItem::new(Line::from(Span::styled(
            summary,
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))));
    }

    let list = List::new(items);
    f.render_widget(list, inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::scheduler::PhaseRecord;
    use crate::bench::slo::{SloEvaluation, SloTier};

    #[test]
    fn fmt_dur_formats_mmss() {
        assert_eq!(fmt_dur(Duration::from_secs(0)), "00:00");
        assert_eq!(fmt_dur(Duration::from_secs(45)), "00:45");
        assert_eq!(fmt_dur(Duration::from_secs(60)), "01:00");
        assert_eq!(fmt_dur(Duration::from_secs(3 * 60 + 21)), "03:21");
    }

    #[test]
    fn state_applies_events_in_order() {
        let tier = SloTier::from_id(2).unwrap();
        let mut state = TuiState::default();
        state.apply(Event::Started {
            tier,
            total_max_users: 256,
        });
        assert_eq!(state.tier, Some(tier));
        assert_eq!(state.max_users_cap, 256);

        state.apply(Event::PhaseStarted {
            phase_index: 1,
            concurrent_users: 4,
            kind: "ramp".into(),
            warmup: Duration::from_secs(30),
            steady_state: Duration::from_secs(30),
        });
        assert_eq!(state.current_phase_index, 1);
        assert_eq!(state.current_phase_users, 4);
        assert!(state.in_warmup);

        state.apply(Event::PhaseProgress {
            phase_index: 1,
            elapsed_in_phase: Duration::from_secs(35),
            in_warmup: false,
            steady_state_samples_so_far: 12,
            running_p25_speed: 88.0,
            running_p95_ttft: 1.1,
            running_system_throughput_tok_per_s: 250.0,
            active_requests: 4,
        });
        assert!(!state.in_warmup);
        assert_eq!(state.steady_state_samples, 12);
        assert!((state.running_p25_speed - 88.0).abs() < 1e-9);

        let phase = PhaseRecord {
            phase_index: 1,
            concurrent_users: 4,
            kind: "ramp".into(),
            steady_state_samples: 30,
            evaluation: SloEvaluation::evaluate(80.0, 1.1, tier),
            duration_seconds: 60.0,
        };
        state.apply(Event::PhaseFinished { record: phase });
        assert_eq!(state.history.len(), 1);

        let report = ScheduleReport {
            tier,
            max_users_explored: 4,
            max_users_cap: 256,
            saturation_users: Some(4),
            phases: state.history.clone(),
            total_wall_seconds: 60.0,
        };
        state.apply(Event::Finished { report });
        assert!(state.finished);
        assert!(state.final_report.is_some());
    }
}
