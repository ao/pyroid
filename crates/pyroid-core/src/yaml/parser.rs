// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! The parser (`parser.py`): tokens to events, the same state machine.

use super::error::{ErrorKind, Mark, Msg, Result, YamlError};
use super::events::{Event, EventKind};
use super::scanner::Scanner;
use super::tokens::{DirectiveValue, T, Token, TokenKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    StreamStart,
    ImplicitDocumentStart,
    DocumentStart,
    DocumentEnd,
    DocumentContent,
    BlockNode,
    BlockSequenceFirstEntry,
    BlockSequenceEntry,
    IndentlessSequenceEntry,
    BlockMappingFirstKey,
    BlockMappingKey,
    BlockMappingValue,
    FlowSequenceFirstEntry,
    FlowSequenceEntry,
    FlowSequenceEntryMappingKey,
    FlowSequenceEntryMappingValue,
    FlowSequenceEntryMappingEnd,
    FlowMappingFirstKey,
    FlowMappingKey,
    FlowMappingValue,
    FlowMappingEmptyValue,
}

const DEFAULT_TAGS: [(&str, &str); 2] = [("!", "!"), ("!!", "tag:yaml.org,2002:")];

fn parser_error(
    context: Option<Msg>,
    context_mark: Option<Mark>,
    problem: Msg,
    problem_mark: Mark,
) -> YamlError {
    YamlError::marked(
        ErrorKind::Parser,
        context,
        context_mark,
        problem,
        Some(problem_mark),
    )
}

/// PyYAML's `Parser` over a [`Scanner`].
#[derive(Debug)]
pub struct Parser {
    pub scanner: Scanner,
    current_event: Option<Event>,
    tag_handles: Vec<(String, String)>,
    states: Vec<State>,
    marks: Vec<Mark>,
    state: Option<State>,
}

impl Parser {
    pub fn new(scanner: Scanner) -> Self {
        Self {
            scanner,
            current_event: None,
            tag_handles: Vec::new(),
            states: Vec::new(),
            marks: Vec::new(),
            state: Some(State::StreamStart),
        }
    }

    /// `peek_event()` (`None` after the stream end).
    pub fn peek_event(&mut self) -> Result<Option<&Event>> {
        if self.current_event.is_none()
            && let Some(state) = self.state
        {
            self.current_event = Some(self.run(state)?);
        }
        Ok(self.current_event.as_ref())
    }

    /// `get_event()`.
    pub fn get_event(&mut self) -> Result<Option<Event>> {
        self.peek_event()?;
        Ok(self.current_event.take())
    }

    fn run(&mut self, state: State) -> Result<Event> {
        match state {
            State::StreamStart => self.parse_stream_start(),
            State::ImplicitDocumentStart => self.parse_implicit_document_start(),
            State::DocumentStart => self.parse_document_start(),
            State::DocumentEnd => self.parse_document_end(),
            State::DocumentContent => self.parse_document_content(),
            State::BlockNode => self.parse_node(true, false),
            State::BlockSequenceFirstEntry => {
                let token = self.get_token()?;
                self.marks.push(token.start_mark);
                self.parse_block_sequence_entry()
            }
            State::BlockSequenceEntry => self.parse_block_sequence_entry(),
            State::IndentlessSequenceEntry => self.parse_indentless_sequence_entry(),
            State::BlockMappingFirstKey => {
                let token = self.get_token()?;
                self.marks.push(token.start_mark);
                self.parse_block_mapping_key()
            }
            State::BlockMappingKey => self.parse_block_mapping_key(),
            State::BlockMappingValue => self.parse_block_mapping_value(),
            State::FlowSequenceFirstEntry => {
                let token = self.get_token()?;
                self.marks.push(token.start_mark);
                self.parse_flow_sequence_entry(true)
            }
            State::FlowSequenceEntry => self.parse_flow_sequence_entry(false),
            State::FlowSequenceEntryMappingKey => self.parse_flow_sequence_entry_mapping_key(),
            State::FlowSequenceEntryMappingValue => self.parse_flow_sequence_entry_mapping_value(),
            State::FlowSequenceEntryMappingEnd => {
                self.state = Some(State::FlowSequenceEntry);
                let mark = self.peek_token()?.start_mark;
                Ok(event(EventKind::MappingEnd, mark, mark))
            }
            State::FlowMappingFirstKey => {
                let token = self.get_token()?;
                self.marks.push(token.start_mark);
                self.parse_flow_mapping_key(true)
            }
            State::FlowMappingKey => self.parse_flow_mapping_key(false),
            State::FlowMappingValue => self.parse_flow_mapping_value(),
            State::FlowMappingEmptyValue => {
                self.state = Some(State::FlowMappingKey);
                let mark = self.peek_token()?.start_mark;
                Ok(empty_scalar(mark))
            }
        }
    }

    // Token helpers. PyYAML's peek_token() never returns None where the parser calls it (the
    // stream always ends with STREAM-END, after which the parser stops).

    #[inline]
    fn check(&mut self, choices: &[T]) -> Result<bool> {
        self.scanner.check_token(choices)
    }

    #[inline]
    fn peek_token(&mut self) -> Result<&Token> {
        match self.scanner.peek_token()? {
            Some(t) => Ok(t),
            None => Err(YamlError::Value(
                "parser read past the end of the stream".to_owned(),
            )),
        }
    }

    #[inline]
    fn get_token(&mut self) -> Result<Token> {
        match self.scanner.get_token()? {
            Some(t) => Ok(t),
            None => Err(YamlError::Value(
                "parser read past the end of the stream".to_owned(),
            )),
        }
    }

    #[inline]
    fn pop_state(&mut self) {
        self.state = self.states.pop();
    }

    fn parse_stream_start(&mut self) -> Result<Event> {
        let token = self.get_token()?;
        self.state = Some(State::ImplicitDocumentStart);
        Ok(event(
            EventKind::StreamStart { encoding: None },
            token.start_mark,
            token.end_mark,
        ))
    }

    fn parse_implicit_document_start(&mut self) -> Result<Event> {
        if self.check(&[T::Directive, T::DocumentStart, T::StreamEnd])? {
            return self.parse_document_start();
        }
        self.tag_handles = DEFAULT_TAGS
            .iter()
            .map(|&(h, p)| (h.to_owned(), p.to_owned()))
            .collect();
        let mark = self.peek_token()?.start_mark;
        self.states.push(State::DocumentEnd);
        self.state = Some(State::BlockNode);
        Ok(event(
            EventKind::DocumentStart {
                explicit: false,
                version: None,
                tags: None,
            },
            mark,
            mark,
        ))
    }

    fn parse_document_start(&mut self) -> Result<Event> {
        while self.check(&[T::DocumentEnd])? {
            self.get_token()?;
        }
        if self.check(&[T::StreamEnd])? {
            let token = self.get_token()?;
            self.state = None;
            return Ok(event(
                EventKind::StreamEnd,
                token.start_mark,
                token.end_mark,
            ));
        }
        let start_mark = self.peek_token()?.start_mark;
        let (version, tags) = self.process_directives()?;
        if !self.check(&[T::DocumentStart])? {
            let token = self.peek_token()?;
            return Err(parser_error(
                None,
                None,
                Msg::str("expected '<document start>', but found %r", token.kind.id()),
                token.start_mark,
            ));
        }
        let token = self.get_token()?;
        self.states.push(State::DocumentEnd);
        self.state = Some(State::DocumentContent);
        Ok(event(
            EventKind::DocumentStart {
                explicit: true,
                version,
                tags,
            },
            start_mark,
            token.end_mark,
        ))
    }

    fn parse_document_end(&mut self) -> Result<Event> {
        let start_mark = self.peek_token()?.start_mark;
        let mut end_mark = start_mark;
        let mut explicit = false;
        if self.check(&[T::DocumentEnd])? {
            end_mark = self.get_token()?.end_mark;
            explicit = true;
        }
        self.state = Some(State::DocumentStart);
        Ok(event(
            EventKind::DocumentEnd { explicit },
            start_mark,
            end_mark,
        ))
    }

    fn parse_document_content(&mut self) -> Result<Event> {
        if self.check(&[T::Directive, T::DocumentStart, T::DocumentEnd, T::StreamEnd])? {
            let mark = self.peek_token()?.start_mark;
            self.pop_state();
            return Ok(empty_scalar(mark));
        }
        self.parse_node(true, false)
    }

    #[allow(clippy::type_complexity)]
    fn process_directives(
        &mut self,
    ) -> Result<(Option<(u64, u64)>, Option<Vec<(String, String)>>)> {
        let mut yaml_version: Option<(u64, u64)> = None;
        self.tag_handles.clear();
        while self.check(&[T::Directive])? {
            let token = self.get_token()?;
            if let TokenKind::Directive { name, value } = token.kind {
                if name == "YAML" {
                    if yaml_version.is_some() {
                        return Err(parser_error(
                            None,
                            None,
                            Msg::text("found duplicate YAML directive"),
                            token.start_mark,
                        ));
                    }
                    let DirectiveValue::Yaml(major, minor) = value else {
                        unreachable!()
                    };
                    if major != 1 {
                        return Err(parser_error(
                            None,
                            None,
                            Msg::text("found incompatible YAML document (version 1.* is required)"),
                            token.start_mark,
                        ));
                    }
                    yaml_version = Some((major, minor));
                } else if name == "TAG" {
                    let DirectiveValue::Tag(handle, prefix) = value else {
                        unreachable!()
                    };
                    if self.tag_handles.iter().any(|(h, _)| *h == handle) {
                        return Err(parser_error(
                            None,
                            None,
                            Msg::str("duplicate tag handle %r", handle),
                            token.start_mark,
                        ));
                    }
                    self.tag_handles.push((handle, prefix));
                }
            }
        }
        let tags = if self.tag_handles.is_empty() {
            None
        } else {
            let mut t = self.tag_handles.clone();
            t.sort();
            Some(t)
        };
        for (h, p) in DEFAULT_TAGS {
            if !self.tag_handles.iter().any(|(k, _)| k == h) {
                self.tag_handles.push((h.to_owned(), p.to_owned()));
            }
        }
        Ok((yaml_version, tags))
    }

    #[allow(clippy::too_many_lines)] // PyYAML's parse_node, kept in one piece
    fn parse_node(&mut self, block: bool, indentless_sequence: bool) -> Result<Event> {
        if self.check(&[T::Alias])? {
            let token = self.get_token()?;
            let TokenKind::Alias(anchor) = token.kind else {
                unreachable!()
            };
            self.pop_state();
            return Ok(event(
                EventKind::Alias { anchor },
                token.start_mark,
                token.end_mark,
            ));
        }
        let mut anchor = None;
        let mut tag_pair: Option<(Option<String>, String)> = None;
        let mut start_mark: Option<Mark> = None;
        let mut end_mark: Option<Mark> = None;
        let mut tag_mark: Option<Mark> = None;
        if self.check(&[T::Anchor])? {
            let token = self.get_token()?;
            start_mark = Some(token.start_mark);
            end_mark = Some(token.end_mark);
            let TokenKind::Anchor(a) = token.kind else {
                unreachable!()
            };
            anchor = Some(a);
            if self.check(&[T::Tag])? {
                let token = self.get_token()?;
                tag_mark = Some(token.start_mark);
                end_mark = Some(token.end_mark);
                let TokenKind::Tag(h, s) = token.kind else {
                    unreachable!()
                };
                tag_pair = Some((h, s));
            }
        } else if self.check(&[T::Tag])? {
            let token = self.get_token()?;
            start_mark = Some(token.start_mark);
            tag_mark = Some(token.start_mark);
            end_mark = Some(token.end_mark);
            let TokenKind::Tag(h, s) = token.kind else {
                unreachable!()
            };
            tag_pair = Some((h, s));
            if self.check(&[T::Anchor])? {
                let token = self.get_token()?;
                end_mark = Some(token.end_mark);
                let TokenKind::Anchor(a) = token.kind else {
                    unreachable!()
                };
                anchor = Some(a);
            }
        }
        let tag: Option<String> = match tag_pair {
            None => None,
            Some((None, suffix)) => Some(suffix),
            Some((Some(handle), suffix)) => {
                match self.tag_handles.iter().find(|(h, _)| *h == handle) {
                    Some((_, prefix)) => Some(format!("{prefix}{suffix}")),
                    None => {
                        return Err(parser_error(
                            Some(Msg::text("while parsing a node")),
                            start_mark,
                            Msg::str("found undefined tag handle %r", handle),
                            tag_mark.unwrap_or_default(),
                        ));
                    }
                }
            }
        };
        let (start_mark, mut end_mark) = if let Some(s) = start_mark {
            (s, end_mark.unwrap_or(s))
        } else {
            let m = self.peek_token()?.start_mark;
            (m, m)
        };
        let implicit = tag.is_none() || tag.as_deref() == Some("!");
        if indentless_sequence && self.check(&[T::BlockEntry])? {
            end_mark = self.peek_token()?.end_mark;
            self.state = Some(State::IndentlessSequenceEntry);
            return Ok(event(
                EventKind::SequenceStart {
                    anchor,
                    tag,
                    implicit,
                    flow_style: None,
                },
                start_mark,
                end_mark,
            ));
        }
        if self.check(&[T::Scalar])? {
            let token = self.get_token()?;
            let TokenKind::Scalar {
                value,
                plain,
                style,
            } = token.kind
            else {
                unreachable!()
            };
            let implicit = if (plain && tag.is_none()) || tag.as_deref() == Some("!") {
                (true, false)
            } else if tag.is_none() {
                (false, true)
            } else {
                (false, false)
            };
            self.pop_state();
            return Ok(event(
                EventKind::Scalar {
                    anchor,
                    tag,
                    implicit,
                    value,
                    style,
                },
                start_mark,
                token.end_mark,
            ));
        }
        if self.check(&[T::FlowSequenceStart])? {
            end_mark = self.peek_token()?.end_mark;
            self.state = Some(State::FlowSequenceFirstEntry);
            return Ok(event(
                EventKind::SequenceStart {
                    anchor,
                    tag,
                    implicit,
                    flow_style: Some(true),
                },
                start_mark,
                end_mark,
            ));
        }
        if self.check(&[T::FlowMappingStart])? {
            end_mark = self.peek_token()?.end_mark;
            self.state = Some(State::FlowMappingFirstKey);
            return Ok(event(
                EventKind::MappingStart {
                    anchor,
                    tag,
                    implicit,
                    flow_style: Some(true),
                },
                start_mark,
                end_mark,
            ));
        }
        if block && self.check(&[T::BlockSequenceStart])? {
            end_mark = self.peek_token()?.start_mark;
            self.state = Some(State::BlockSequenceFirstEntry);
            return Ok(event(
                EventKind::SequenceStart {
                    anchor,
                    tag,
                    implicit,
                    flow_style: Some(false),
                },
                start_mark,
                end_mark,
            ));
        }
        if block && self.check(&[T::BlockMappingStart])? {
            end_mark = self.peek_token()?.start_mark;
            self.state = Some(State::BlockMappingFirstKey);
            return Ok(event(
                EventKind::MappingStart {
                    anchor,
                    tag,
                    implicit,
                    flow_style: Some(false),
                },
                start_mark,
                end_mark,
            ));
        }
        if anchor.is_some() || tag.is_some() {
            self.pop_state();
            return Ok(event(
                EventKind::Scalar {
                    anchor,
                    tag,
                    implicit: (implicit, false),
                    value: String::new(),
                    style: None,
                },
                start_mark,
                end_mark,
            ));
        }
        let token = self.peek_token()?;
        let context = if block {
            "while parsing a block node"
        } else {
            "while parsing a flow node"
        };
        Err(parser_error(
            Some(Msg::text(context)),
            Some(start_mark),
            Msg::str("expected the node content, but found %r", token.kind.id()),
            token.start_mark,
        ))
    }

    fn parse_block_sequence_entry(&mut self) -> Result<Event> {
        if self.check(&[T::BlockEntry])? {
            let token = self.get_token()?;
            if self.check(&[T::BlockEntry, T::BlockEnd])? {
                self.state = Some(State::BlockSequenceEntry);
                return Ok(empty_scalar(token.end_mark));
            }
            self.states.push(State::BlockSequenceEntry);
            return self.parse_node(true, false);
        }
        if !self.check(&[T::BlockEnd])? {
            let mark = self.marks.last().copied();
            let token = self.peek_token()?;
            return Err(parser_error(
                Some(Msg::text("while parsing a block collection")),
                mark,
                Msg::str("expected <block end>, but found %r", token.kind.id()),
                token.start_mark,
            ));
        }
        let token = self.get_token()?;
        self.pop_state();
        self.marks.pop();
        Ok(event(
            EventKind::SequenceEnd,
            token.start_mark,
            token.end_mark,
        ))
    }

    fn parse_indentless_sequence_entry(&mut self) -> Result<Event> {
        if self.check(&[T::BlockEntry])? {
            let token = self.get_token()?;
            if self.check(&[T::BlockEntry, T::Key, T::Value, T::BlockEnd])? {
                self.state = Some(State::IndentlessSequenceEntry);
                return Ok(empty_scalar(token.end_mark));
            }
            self.states.push(State::IndentlessSequenceEntry);
            return self.parse_node(true, false);
        }
        let mark = self.peek_token()?.start_mark;
        self.pop_state();
        Ok(event(EventKind::SequenceEnd, mark, mark))
    }

    fn parse_block_mapping_key(&mut self) -> Result<Event> {
        if self.check(&[T::Key])? {
            let token = self.get_token()?;
            if self.check(&[T::Key, T::Value, T::BlockEnd])? {
                self.state = Some(State::BlockMappingValue);
                return Ok(empty_scalar(token.end_mark));
            }
            self.states.push(State::BlockMappingValue);
            return self.parse_node(true, true);
        }
        if !self.check(&[T::BlockEnd])? {
            let mark = self.marks.last().copied();
            let token = self.peek_token()?;
            return Err(parser_error(
                Some(Msg::text("while parsing a block mapping")),
                mark,
                Msg::str("expected <block end>, but found %r", token.kind.id()),
                token.start_mark,
            ));
        }
        let token = self.get_token()?;
        self.pop_state();
        self.marks.pop();
        Ok(event(
            EventKind::MappingEnd,
            token.start_mark,
            token.end_mark,
        ))
    }

    fn parse_block_mapping_value(&mut self) -> Result<Event> {
        if self.check(&[T::Value])? {
            let token = self.get_token()?;
            if self.check(&[T::Key, T::Value, T::BlockEnd])? {
                self.state = Some(State::BlockMappingKey);
                return Ok(empty_scalar(token.end_mark));
            }
            self.states.push(State::BlockMappingKey);
            return self.parse_node(true, true);
        }
        self.state = Some(State::BlockMappingKey);
        let mark = self.peek_token()?.start_mark;
        Ok(empty_scalar(mark))
    }

    fn parse_flow_sequence_entry(&mut self, first: bool) -> Result<Event> {
        if !self.check(&[T::FlowSequenceEnd])? {
            if !first {
                if self.check(&[T::FlowEntry])? {
                    self.get_token()?;
                } else {
                    let mark = self.marks.last().copied();
                    let token = self.peek_token()?;
                    return Err(parser_error(
                        Some(Msg::text("while parsing a flow sequence")),
                        mark,
                        Msg::str("expected ',' or ']', but got %r", token.kind.id()),
                        token.start_mark,
                    ));
                }
            }
            if self.check(&[T::Key])? {
                let token = self.peek_token()?;
                let (s, e) = (token.start_mark, token.end_mark);
                self.state = Some(State::FlowSequenceEntryMappingKey);
                return Ok(event(
                    EventKind::MappingStart {
                        anchor: None,
                        tag: None,
                        implicit: true,
                        flow_style: Some(true),
                    },
                    s,
                    e,
                ));
            } else if !self.check(&[T::FlowSequenceEnd])? {
                self.states.push(State::FlowSequenceEntry);
                return self.parse_node(false, false);
            }
        }
        let token = self.get_token()?;
        self.pop_state();
        self.marks.pop();
        Ok(event(
            EventKind::SequenceEnd,
            token.start_mark,
            token.end_mark,
        ))
    }

    fn parse_flow_sequence_entry_mapping_key(&mut self) -> Result<Event> {
        let token = self.get_token()?;
        if self.check(&[T::Value, T::FlowEntry, T::FlowSequenceEnd])? {
            self.state = Some(State::FlowSequenceEntryMappingValue);
            return Ok(empty_scalar(token.end_mark));
        }
        self.states.push(State::FlowSequenceEntryMappingValue);
        self.parse_node(false, false)
    }

    fn parse_flow_sequence_entry_mapping_value(&mut self) -> Result<Event> {
        if self.check(&[T::Value])? {
            let token = self.get_token()?;
            if self.check(&[T::FlowEntry, T::FlowSequenceEnd])? {
                self.state = Some(State::FlowSequenceEntryMappingEnd);
                return Ok(empty_scalar(token.end_mark));
            }
            self.states.push(State::FlowSequenceEntryMappingEnd);
            return self.parse_node(false, false);
        }
        self.state = Some(State::FlowSequenceEntryMappingEnd);
        let mark = self.peek_token()?.start_mark;
        Ok(empty_scalar(mark))
    }

    fn parse_flow_mapping_key(&mut self, first: bool) -> Result<Event> {
        if !self.check(&[T::FlowMappingEnd])? {
            if !first {
                if self.check(&[T::FlowEntry])? {
                    self.get_token()?;
                } else {
                    let mark = self.marks.last().copied();
                    let token = self.peek_token()?;
                    return Err(parser_error(
                        Some(Msg::text("while parsing a flow mapping")),
                        mark,
                        Msg::str("expected ',' or '}', but got %r", token.kind.id()),
                        token.start_mark,
                    ));
                }
            }
            if self.check(&[T::Key])? {
                let token = self.get_token()?;
                if self.check(&[T::Value, T::FlowEntry, T::FlowMappingEnd])? {
                    self.state = Some(State::FlowMappingValue);
                    return Ok(empty_scalar(token.end_mark));
                }
                self.states.push(State::FlowMappingValue);
                return self.parse_node(false, false);
            } else if !self.check(&[T::FlowMappingEnd])? {
                self.states.push(State::FlowMappingEmptyValue);
                return self.parse_node(false, false);
            }
        }
        let token = self.get_token()?;
        self.pop_state();
        self.marks.pop();
        Ok(event(
            EventKind::MappingEnd,
            token.start_mark,
            token.end_mark,
        ))
    }

    fn parse_flow_mapping_value(&mut self) -> Result<Event> {
        if self.check(&[T::Value])? {
            let token = self.get_token()?;
            if self.check(&[T::FlowEntry, T::FlowMappingEnd])? {
                self.state = Some(State::FlowMappingKey);
                return Ok(empty_scalar(token.end_mark));
            }
            self.states.push(State::FlowMappingKey);
            return self.parse_node(false, false);
        }
        self.state = Some(State::FlowMappingKey);
        let mark = self.peek_token()?.start_mark;
        Ok(empty_scalar(mark))
    }
}

#[inline]
fn event(kind: EventKind, start_mark: Mark, end_mark: Mark) -> Event {
    Event {
        kind,
        start_mark,
        end_mark,
    }
}

#[inline]
fn empty_scalar(mark: Mark) -> Event {
    event(
        EventKind::Scalar {
            anchor: None,
            tag: None,
            implicit: (true, false),
            value: String::new(),
            style: None,
        },
        mark,
        mark,
    )
}
