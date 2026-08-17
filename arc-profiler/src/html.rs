//! Renders one or more [`Profile`]s into a **single self-contained** HTML page.
//!
//! No CDN, no external stylesheet, no font, no fetch: the page has to open from
//! a laptop with no network, months after the box that produced it was deleted.
//! The template is compiled into the binary and the profile JSON is spliced
//! into it at one marker.

use crate::report::Profile;

/// The marker the template reserves for the data. Kept as a JS comment plus an
/// empty array so the template is *also* valid, openable HTML on its own.
const MARKER: &str = "/*__ARC_PROFILE_DATA__*/[]";

const TEMPLATE: &str = include_str!("template.html");

/// Splice `profiles` into the template.
///
/// If the marker is missing the data is appended in a trailing `<script>`
/// instead of silently producing a page with no data — a report that renders
/// empty looks like "the profiler found nothing", which is the wrong answer.
pub fn render(profiles: &[Profile]) -> String {
    let data = serde_json::to_string(profiles)
        .unwrap_or_else(|e| format!("[{{\"error\":\"{}\"}}]", e.to_string().replace('"', "'")));
    if TEMPLATE.contains(MARKER) {
        TEMPLATE.replacen(MARKER, &data, 1)
    } else {
        format!("{TEMPLATE}\n<script>window.RUNS = {data};</script>\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_carries_the_data_marker() {
        // If someone edits the template and drops the marker, `render` silently
        // falls back to an appended script that the page's own JS never reads.
        // Fail here instead.
        assert!(
            TEMPLATE.contains(MARKER),
            "template.html must contain the literal `{MARKER}`"
        );
    }

    #[test]
    fn render_embeds_the_profiles_and_no_network_references() {
        let p = crate::snapshot();
        let out = render(std::slice::from_ref(&p));
        assert!(out.contains("arc-profile/1"), "schema must be embedded");
        assert!(!out.contains(MARKER), "marker must be consumed");
        for forbidden in ["http://", "https://", "//cdn.", "src=\"/", "@import url("] {
            assert!(
                !out.contains(forbidden),
                "page must be self-contained; found `{forbidden}`"
            );
        }
    }
}
