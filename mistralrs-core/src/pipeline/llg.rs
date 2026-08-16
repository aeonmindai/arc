use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use llguidance::{api::TopLevelGrammar, ParserFactory};
use tokenizers::Tokenizer;

use crate::Constraint;

/// Give bytes back to tokens that `ByteTokenizer::from_tokenizer` gave up on.
///
/// For a byte-level tokenizer, `from_tokenizer` maps each char of a vocab entry
/// through the GPT-2 byte<->char table. A char outside that table makes it log
/// `error: missing char: <c> for "<token>"` and `continue`
/// (toktrie_hf_tokenizers-1.7.0/src/lib.rs:205-212), which leaves that id's
/// entry as the empty Vec it was initialised to at lib.rs:150.
///
/// An empty entry is not inert. `TokTrie::decode_ext`
/// (toktrie-1.7.0/src/toktree.rs:451-458) turns it into *nothing* when
/// `include_special` is false, and every decode path in this engine runs
/// through it — `finish_or_add_toks_to_seq` detokenizes each sampled token
/// there and appends the result to `Sequence::completion_bytes`
/// (pipeline/sampling.rs:45-48). So a broken id is silently deleted from the
/// response text, splicing whatever came before it onto whatever came after
/// with no separator, and it can never contribute to a stop-STRING match,
/// which scans exactly those bytes (sequence.rs:1003-1013).
///
/// DeepSeek-V4 trips this on every load — `missing char: ｜ for "｜DSML｜"` and
/// `for "<｜/table>｜"` — because U+FF5C FULLWIDTH VERTICAL LINE is not in the
/// byte<->char table and those tokens live in the base BPE vocab, so the
/// `added_special` repair above does not reach them. `｜DSML｜` is the delimiter
/// V4 builds every tool call out of, so leaving it empty is not cosmetic.
///
/// The repair is deliberately narrow: entries that already carry bytes are
/// left exactly as they are, so nothing that decodes correctly today changes.
/// Returns the number of ids repaired.
fn repair_missing_token_bytes(
    token_bytes: &mut [Vec<u8>],
    vocab: &HashMap<String, u32>,
) -> Vec<String> {
    // Deterministic order: a HashMap walk would make the repair set depend on
    // hash seed if a vocab ever mapped two strings to one id.
    let mut entries: Vec<(&String, &u32)> = vocab.iter().collect();
    entries.sort_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)));

    let mut repaired = Vec::new();
    for (content, id) in entries {
        let idx = *id as usize;
        if idx < token_bytes.len() && token_bytes[idx].is_empty() && !content.is_empty() {
            token_bytes[idx] = content.as_bytes().to_vec();
            repaired.push(content.clone());
        }
    }
    repaired
}

pub fn build_llg_factory(tokenizer: Tokenizer) -> Result<Arc<ParserFactory>> {
    // Collect special token info before from_tokenizer() consumes the tokenizer.
    let added_special: Vec<(u32, String)> = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, at)| at.special)
        .map(|(id, at)| (id, at.content))
        .collect();
    let vocab = tokenizer.get_vocab(true);

    let bt = toktrie_hf_tokenizers::ByteTokenizer::from_tokenizer(tokenizer)?;
    let info = bt.tokrx_info();
    let mut token_bytes = bt.token_bytes();

    // Fix up special tokens that from_tokenizer() may have missed.
    // Tekken tokenizers add special tokens to the BPE base vocab first, which can
    // prevent the toktrie builder from applying the SPECIAL_TOKEN_MARKER prefix.
    for (id, content) in &added_special {
        let idx = *id as usize;
        if idx < token_bytes.len()
            && (token_bytes[idx].is_empty()
                || token_bytes[idx][0] != toktrie::TokTrie::SPECIAL_TOKEN_MARKER)
        {
            let mut bytes = content.as_bytes().to_vec();
            bytes.insert(0, toktrie::TokTrie::SPECIAL_TOKEN_MARKER);
            token_bytes[idx] = bytes;
        }
    }

    // Then anything the byte<->char mapping dropped entirely. Runs after the
    // added_special pass so a special token keeps its SPECIAL_TOKEN_MARKER.
    let repaired = repair_missing_token_bytes(&mut token_bytes, &vocab);
    if !repaired.is_empty() {
        let sample: Vec<&str> = repaired.iter().take(8).map(String::as_str).collect();
        tracing::warn!(
            "toktrie: restored bytes for {} vocab token(s) that the byte-level \
             decoder could not map (e.g. {:?}). Without this they decode to the \
             empty string and vanish from generated text.",
            repaired.len(),
            sample
        );
    }

    let tok_trie = toktrie::TokTrie::from(&info, &token_bytes);
    let env = toktrie_hf_tokenizers::ByteTokenizerEnv {
        tokenizer: bt,
        tok_trie,
    }
    .to_env();
    let factory = ParserFactory::new_simple(&env)?;
    Ok(Arc::new(factory))
}

pub fn llg_grammar_from_constraint(constraint: &Constraint) -> Result<Option<TopLevelGrammar>> {
    let grm = match constraint {
        Constraint::Regex(regex) => TopLevelGrammar::from_regex(regex),
        Constraint::Lark(lark) => TopLevelGrammar::from_lark(lark.clone()),
        Constraint::JsonSchema(value) => TopLevelGrammar::from_json_schema(value.clone()),
        Constraint::Llguidance(value) => value.clone(),
        Constraint::None => return Ok(None),
    };
    Ok(Some(grm))
}

pub fn constraint_from_llg_grammar(
    factory: &ParserFactory,
    grm: TopLevelGrammar,
) -> Result<llguidance::Matcher> {
    let parser = factory.create_parser(grm)?;
    Ok(llguidance::Matcher::new(Ok(parser)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MARKER: u8 = toktrie::TokTrie::SPECIAL_TOKEN_MARKER;

    /// Vocab shaped like the real DeepSeek-V4 failure:
    ///   id 0 "Hello"     - decoded fine
    ///   id 1 "Ġthe"      - decoded fine, and its BYTES DIFFER from its vocab
    ///                      string (byte-level "Ġ" is really a space). This is
    ///                      the discriminating entry: an implementation that
    ///                      overwrote every id from the vocab instead of only
    ///                      the empty ones would corrupt it into "Ġthe", so the
    ///                      fixture can tell the two apart. DOCTRINE D12.
    ///   id 2 "｜DSML｜"    - EMPTY: U+FF5C is not in the byte<->char table
    ///   id 3 "<｜/table>｜" - EMPTY, same reason
    ///   id 4 "<|eos|>"   - already carries SPECIAL_TOKEN_MARKER
    ///   id 5             - not in the vocab at all
    fn fixture() -> (Vec<Vec<u8>>, HashMap<String, u32>) {
        let token_bytes = vec![
            b"Hello".to_vec(),
            b" the".to_vec(),
            Vec::new(),
            Vec::new(),
            {
                let mut v = vec![MARKER];
                v.extend_from_slice(b"<|eos|>");
                v
            },
            Vec::new(),
        ];
        let vocab: HashMap<String, u32> = [
            ("Hello".to_string(), 0u32),
            ("Ġthe".to_string(), 1),
            ("｜DSML｜".to_string(), 2),
            ("<｜/table>｜".to_string(), 3),
            ("<|eos|>".to_string(), 4),
        ]
        .into_iter()
        .collect();
        (token_bytes, vocab)
    }

    #[test]
    fn fixture_discriminates() {
        // If id 1's stored bytes equalled its vocab string, "fill only the
        // empties" and "overwrite everything" would be indistinguishable and
        // every assertion below would be vacuous. Assert they differ first.
        let (token_bytes, vocab) = fixture();
        let name = vocab.iter().find(|(_, &v)| v == 1).unwrap().0;
        assert_ne!(
            token_bytes[1].as_slice(),
            name.as_bytes(),
            "fixture is vacuous: id 1's bytes must differ from its vocab string"
        );
        // And the broken ids must actually start empty.
        assert!(token_bytes[2].is_empty() && token_bytes[3].is_empty());
    }

    #[test]
    fn repairs_only_empty_entries() {
        let (mut token_bytes, vocab) = fixture();
        let before = token_bytes.clone();
        let repaired = repair_missing_token_bytes(&mut token_bytes, &vocab);

        // The two DeepSeek-style tokens get their UTF-8 bytes back...
        assert_eq!(token_bytes[2], "｜DSML｜".as_bytes());
        assert_eq!(token_bytes[3], "<｜/table>｜".as_bytes());
        assert_eq!(
            repaired,
            vec!["｜DSML｜".to_string(), "<｜/table>｜".to_string()],
            "repaired ids must be reported in id order"
        );

        // ...and nothing that already decoded is touched. This is the promise
        // that ordinary generations decode exactly as they did before.
        assert_eq!(token_bytes[0], before[0]);
        assert_eq!(token_bytes[1], before[1], "byte-level bytes must survive");
        assert_eq!(
            token_bytes[4], before[4],
            "SPECIAL_TOKEN_MARKER must survive"
        );

        // An id with no vocab entry stays empty rather than being invented.
        assert!(token_bytes[5].is_empty());
    }

    #[test]
    fn repair_is_idempotent() {
        let (mut token_bytes, vocab) = fixture();
        assert_eq!(
            repair_missing_token_bytes(&mut token_bytes, &vocab).len(),
            2
        );
        let after_first = token_bytes.clone();
        assert!(
            repair_missing_token_bytes(&mut token_bytes, &vocab).is_empty(),
            "second pass must find nothing left to repair"
        );
        assert_eq!(token_bytes, after_first);
    }

    #[test]
    fn healthy_vocab_is_left_alone() {
        // The no-op case: every id already has bytes, so the repair must be a
        // pure identity. If this ever fails, the change is not additive.
        let mut token_bytes = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let before = token_bytes.clone();
        let vocab: HashMap<String, u32> = [
            ("a".to_string(), 0u32),
            ("b".to_string(), 1),
            ("c".to_string(), 2),
        ]
        .into_iter()
        .collect();
        assert!(repair_missing_token_bytes(&mut token_bytes, &vocab).is_empty());
        assert_eq!(token_bytes, before);
    }

    #[test]
    fn out_of_range_and_empty_names_are_ignored() {
        let mut token_bytes = vec![Vec::new()];
        let vocab: HashMap<String, u32> = [
            ("".to_string(), 0u32), // empty name: nothing to restore
            ("x".to_string(), 99),  // id past the end of token_bytes
        ]
        .into_iter()
        .collect();
        assert!(repair_missing_token_bytes(&mut token_bytes, &vocab).is_empty());
        assert_eq!(token_bytes.len(), 1);
        assert!(token_bytes[0].is_empty());
    }
}
