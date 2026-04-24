use std::collections::VecDeque;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::cursor;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::openai::{self, Input, InputItem, ResponsesRequest};

const DEFAULT_COMPOSE_HEIGHT: usize = 4;

const MODEL_ALIASES: [ModelAlias; 5] = [
    ModelAlias {
        name: "latest",
        model: "gpt-5.5",
    },
    ModelAlias {
        name: "full",
        model: "gpt-5.4",
    },
    ModelAlias {
        name: "mini",
        model: "gpt-5.4-mini",
    },
    ModelAlias {
        name: "nano",
        model: "gpt-5.4-nano",
    },
    ModelAlias {
        name: "default",
        model: openai::DEFAULT_MODEL,
    },
];

#[derive(Clone, Copy)]
struct ModelAlias {
    name: &'static str,
    model: &'static str,
}

#[derive(Debug, Deserialize, Serialize)]
struct TuiConfig {
    model: String,
}

#[derive(Clone, Debug, Default)]
pub struct Options {
    pub initial_prompt: Option<String>,
}

#[derive(Clone, Copy)]
struct Theme {
    name: &'static str,
    header: u8,
    accent: u8,
    user: u8,
    agent: u8,
    note: u8,
}

const THEMES: [Theme; 4] = [
    Theme {
        name: "amp",
        header: 255,
        accent: 43,
        user: 39,
        agent: 255,
        note: 245,
    },
    Theme {
        name: "oxide",
        header: 39,
        accent: 44,
        user: 39,
        agent: 81,
        note: 244,
    },
    Theme {
        name: "ember",
        header: 202,
        accent: 208,
        user: 39,
        agent: 79,
        note: 244,
    },
    Theme {
        name: "lagoon",
        header: 45,
        accent: 51,
        user: 39,
        agent: 86,
        note: 244,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    User,
    Agent,
    Note,
}

struct Message {
    role: Role,
    text: String,
}

enum ContextItem {
    UserText(String),
    ResponseItem(Value),
}

impl ContextItem {
    fn as_input_item(&self) -> InputItem {
        match self {
            Self::UserText(text) => InputItem::user_text(text.clone()),
            Self::ResponseItem(value) => InputItem::raw_json_value(value.clone()),
        }
    }
}

struct Job {
    prompt: String,
    model: String,
    target_message: usize,
}

struct ActiveRequest {
    prompt: String,
    model: String,
    target_message: usize,
}

struct WorkerRequest {
    model: String,
    input_items: Vec<InputItem>,
}

struct TurnCompletion {
    output_text: String,
    output_items: Vec<Value>,
}

enum WorkerEvent {
    TextDelta(String),
    Failed(String),
    Completed(TurnCompletion),
}

struct Search {
    query: String,
    original_draft: String,
    original_cursor: usize,
    original_history_index: Option<usize>,
    match_index: Option<usize>,
}

#[derive(Clone, Copy)]
struct WrappedLine {
    start: usize,
    end: usize,
}

struct TranscriptLine {
    role: Role,
    text: String,
    first: bool,
    active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TranscriptStyle {
    prefix: &'static str,
    color: u8,
    italic: bool,
    bold: bool,
}

#[derive(Default)]
struct InputOutcome {
    dirty: bool,
    quit: bool,
}

#[derive(Debug, PartialEq, Eq)]
enum CommandOutcome {
    NotCommand,
    Handled,
    Rejected,
}

#[derive(Debug, PartialEq, Eq)]
enum ModelArg {
    Empty,
    Invalid,
    Selected(String),
}

enum Key {
    Char(char),
    AltChar(char),
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    Delete,
    Backspace,
    PageUp,
    PageDown,
    Enter,
    Tab,
    Esc,
    CtrlA,
    CtrlB,
    CtrlC,
    CtrlD,
    CtrlE,
    CtrlF,
    CtrlJ,
    CtrlK,
    CtrlL,
    CtrlN,
    CtrlP,
    CtrlR,
    CtrlS,
    CtrlU,
    CtrlW,
    CtrlY,
}

#[derive(Clone, Copy)]
struct Size {
    cols: usize,
    rows: usize,
}

struct TerminalGuard;

impl TerminalGuard {
    fn init() -> Result<Self> {
        terminal::enable_raw_mode().context("failed to enable raw mode")?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, cursor::Hide)
            .context("failed to enter alternate screen")?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, cursor::Show, LeaveAlternateScreen);
    }
}

struct App {
    client: openai::Client,
    size: Size,
    render_buf: String,
    notice: String,
    draft: String,
    kill_buffer: String,
    cursor: usize,
    scroll_from_bottom: usize,
    theme_index: usize,
    spinner: usize,
    location_label: Option<String>,
    current_model: String,
    config_path: Option<PathBuf>,
    messages: Vec<Message>,
    context: Vec<ContextItem>,
    queue: VecDeque<Job>,
    history: Vec<String>,
    history_index: Option<usize>,
    history_draft: Option<String>,
    search: Option<Search>,
    worker_tx: Sender<WorkerEvent>,
    worker_rx: Receiver<WorkerEvent>,
    active_request: Option<ActiveRequest>,
}

impl App {
    fn init(opts: Options) -> Result<Self> {
        let client = openai::Client::from_env()?;
        let size = read_terminal_size()?;
        let location_label = build_location_label()?;
        let config_path = default_config_path();
        let current_model = startup_model(config_path.as_deref());
        let (worker_tx, worker_rx) = mpsc::channel();

        let mut app = Self {
            client,
            size,
            render_buf: String::new(),
            notice: String::new(),
            draft: String::new(),
            kill_buffer: String::new(),
            cursor: 0,
            scroll_from_bottom: 0,
            theme_index: 0,
            spinner: 0,
            location_label,
            current_model,
            config_path,
            messages: Vec::new(),
            context: Vec::new(),
            queue: VecDeque::new(),
            history: Vec::new(),
            history_index: None,
            history_draft: None,
            search: None,
            worker_tx,
            worker_rx,
            active_request: None,
        };

        if let Some(prompt) = opts.initial_prompt {
            app.draft.push_str(&prompt);
            app.cursor = app.draft.len();
        }

        Ok(app)
    }

    fn run(&mut self) -> Result<()> {
        self.render()?;
        loop {
            if self.step()? {
                return Ok(());
            }
        }
    }

    fn step(&mut self) -> Result<bool> {
        let mut dirty = self.refresh_size()?;
        let timeout = if self.active_request.is_some() {
            Duration::from_millis(16)
        } else {
            Duration::from_millis(80)
        };

        if event::poll(timeout).context("failed to poll terminal")? {
            match event::read().context("failed to read terminal event")? {
                Event::Key(key_event)
                    if matches!(key_event.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                {
                    if let Some(key) = key_from_event(key_event) {
                        let outcome = self.handle_key(key)?;
                        dirty |= outcome.dirty;
                        if outcome.quit {
                            return Ok(true);
                        }
                    }
                }
                Event::Paste(text) => {
                    self.reset_history_navigation();
                    self.insert_draft_slice(&text);
                    dirty = true;
                }
                Event::Resize(_, _) => {
                    dirty = true;
                }
                _ => {}
            }
        }

        dirty |= self.drain_worker_events()?;
        if dirty {
            self.render()?;
        }
        Ok(false)
    }

    fn refresh_size(&mut self) -> Result<bool> {
        let next = read_terminal_size()?;
        if next.cols == self.size.cols && next.rows == self.size.rows {
            return Ok(false);
        }
        self.size = next;
        Ok(true)
    }

    fn handle_key(&mut self, key: Key) -> Result<InputOutcome> {
        if self.search.is_some() {
            return self.handle_search_key(key);
        }

        let outcome = match key {
            Key::CtrlC => InputOutcome {
                dirty: false,
                quit: true,
            },
            Key::Esc => InputOutcome::default(),
            Key::CtrlL => InputOutcome {
                dirty: true,
                quit: false,
            },
            Key::Left | Key::CtrlB => {
                self.move_cursor_left();
                dirty_only()
            }
            Key::Right | Key::CtrlF => {
                self.move_cursor_right();
                dirty_only()
            }
            Key::Home | Key::CtrlA => {
                self.cursor = self.current_line_start(self.cursor);
                dirty_only()
            }
            Key::End | Key::CtrlE => {
                self.cursor = self.current_line_end(self.cursor);
                dirty_only()
            }
            Key::Up => {
                if self.move_cursor_vertical(-1)? || self.navigate_history_older()? {
                    dirty_only()
                } else {
                    InputOutcome::default()
                }
            }
            Key::Down => {
                if self.move_cursor_vertical(1)? || self.navigate_history_newer()? {
                    dirty_only()
                } else {
                    InputOutcome::default()
                }
            }
            Key::CtrlP => {
                if self.navigate_history_older()? {
                    dirty_only()
                } else {
                    InputOutcome::default()
                }
            }
            Key::CtrlN => {
                if self.navigate_history_newer()? {
                    dirty_only()
                } else {
                    InputOutcome::default()
                }
            }
            Key::PageUp => {
                self.scroll_from_bottom += clamp_usize(self.size.rows / 2, 1, 12);
                dirty_only()
            }
            Key::PageDown => {
                self.scroll_from_bottom =
                    self.scroll_from_bottom
                        .saturating_sub(clamp_usize(self.size.rows / 2, 1, 12));
                dirty_only()
            }
            Key::Delete | Key::CtrlD => {
                self.reset_history_navigation();
                if self.cursor < self.draft.len() {
                    let width = self.next_char_width(self.cursor);
                    self.draft
                        .replace_range(self.cursor..self.cursor + width, "");
                }
                dirty_only()
            }
            Key::Backspace => {
                self.reset_history_navigation();
                if self.cursor > 0 {
                    let start = self.prev_char_boundary(self.cursor);
                    self.draft.replace_range(start..self.cursor, "");
                    self.cursor = start;
                }
                dirty_only()
            }
            Key::CtrlW => {
                self.reset_history_navigation();
                self.delete_backward_word();
                dirty_only()
            }
            Key::CtrlU => {
                self.reset_history_navigation();
                self.kill_to_beginning_of_line();
                dirty_only()
            }
            Key::CtrlK => {
                self.reset_history_navigation();
                self.kill_to_end_of_line();
                dirty_only()
            }
            Key::CtrlY => {
                self.reset_history_navigation();
                let text = self.kill_buffer.clone();
                self.insert_draft_slice(&text);
                dirty_only()
            }
            Key::CtrlJ => {
                self.reset_history_navigation();
                self.insert_draft_slice("\n");
                dirty_only()
            }
            Key::Tab => {
                self.submit_draft(true)?;
                dirty_only()
            }
            Key::Enter => {
                self.submit_draft(false)?;
                dirty_only()
            }
            Key::CtrlR => {
                self.begin_history_search()?;
                dirty_only()
            }
            Key::CtrlS => InputOutcome::default(),
            Key::AltChar(byte) => {
                match byte {
                    'b' | 'B' => self.cursor = self.beginning_of_previous_word(),
                    'f' | 'F' => self.cursor = self.end_of_next_word(),
                    _ => {}
                }
                dirty_only()
            }
            Key::Char(ch) => {
                if !ch.is_control() {
                    self.reset_history_navigation();
                    let mut text = String::new();
                    text.push(ch);
                    self.insert_draft_slice(&text);
                }
                dirty_only()
            }
        };

        Ok(outcome)
    }

    fn handle_search_key(&mut self, key: Key) -> Result<InputOutcome> {
        match key {
            Key::CtrlC | Key::Esc => {
                self.cancel_search();
                Ok(dirty_only())
            }
            Key::Enter => {
                self.accept_search();
                Ok(dirty_only())
            }
            Key::Backspace => {
                if let Some(search) = &mut self.search {
                    truncate_last_codepoint(&mut search.query);
                }
                self.update_search_preview();
                Ok(dirty_only())
            }
            Key::Up | Key::CtrlP | Key::CtrlR => {
                self.search_backward();
                Ok(dirty_only())
            }
            Key::Down | Key::CtrlN | Key::CtrlS => {
                self.search_forward();
                Ok(dirty_only())
            }
            Key::Char(ch) if !ch.is_control() => {
                if let Some(search) = &mut self.search {
                    search.query.push(ch);
                }
                self.update_search_preview();
                Ok(dirty_only())
            }
            _ => Ok(InputOutcome::default()),
        }
    }

    fn insert_draft_slice(&mut self, bytes: &str) {
        self.draft.insert_str(self.cursor, bytes);
        self.cursor += bytes.len();
    }

    fn set_draft_text(&mut self, text: &str) {
        self.draft.clear();
        self.draft.push_str(text);
        self.cursor = self.draft.len();
    }

    fn set_kill_buffer(&mut self, text: &str) {
        self.kill_buffer.clear();
        self.kill_buffer.push_str(text);
    }

    fn append_note(&mut self, text: &str) {
        self.messages.push(Message {
            role: Role::Note,
            text: text.to_owned(),
        });
        self.scroll_from_bottom = 0;
    }

    fn reset_history_navigation(&mut self) {
        self.history_index = None;
        self.history_draft = None;
    }

    fn move_cursor_left(&mut self) {
        self.cursor = self.prev_char_boundary(self.cursor);
    }

    fn move_cursor_right(&mut self) {
        self.cursor = self.next_char_boundary(self.cursor);
    }

    fn move_cursor_vertical(&mut self, delta: isize) -> Result<bool> {
        let width = max_usize(self.size.cols.saturating_sub(3), 1);
        let lines = wrap_lines(&self.draft, width);
        let pos = cursor_visual(&lines, self.cursor);

        if delta < 0 && pos.line == 0 {
            return Ok(false);
        }
        if delta > 0 && pos.line + 1 >= lines.len() {
            return Ok(false);
        }

        let target_line = if delta < 0 {
            pos.line - 1
        } else {
            pos.line + 1
        };
        let span = lines[target_line];
        self.cursor = span.start + min_usize(pos.col, span.end - span.start);
        Ok(true)
    }

    fn current_line_start(&self, pos: usize) -> usize {
        let mut i = min_usize(pos, self.draft.len());
        while i > 0 && self.draft.as_bytes()[i - 1] != b'\n' {
            i -= 1;
        }
        i
    }

    fn current_line_end(&self, pos: usize) -> usize {
        let mut i = min_usize(pos, self.draft.len());
        while i < self.draft.len() && self.draft.as_bytes()[i] != b'\n' {
            i = self.next_char_boundary(i);
        }
        i
    }

    fn navigate_history_older(&mut self) -> Result<bool> {
        if self.history.is_empty() {
            return Ok(false);
        }

        match self.history_index {
            None => {
                self.reset_history_navigation();
                self.history_draft = Some(self.draft.clone());
                self.history_index = Some(self.history.len() - 1);
            }
            Some(0) => return Ok(false),
            Some(index) => self.history_index = Some(index - 1),
        }

        let text = self.history[self.history_index.expect("history index")].clone();
        self.set_draft_text(&text);
        Ok(true)
    }

    fn navigate_history_newer(&mut self) -> Result<bool> {
        let Some(index) = self.history_index else {
            return Ok(false);
        };

        if index + 1 < self.history.len() {
            self.history_index = Some(index + 1);
            let text = self.history[self.history_index.expect("history index")].clone();
            self.set_draft_text(&text);
            return Ok(true);
        }

        let draft = self.history_draft.take();
        self.history_index = None;
        if let Some(text) = draft {
            self.set_draft_text(&text);
        } else {
            self.draft.clear();
            self.cursor = 0;
        }
        Ok(true)
    }

    fn prev_char_boundary(&self, pos: usize) -> usize {
        prev_utf8_boundary(&self.draft, pos)
    }

    fn next_char_boundary(&self, pos: usize) -> usize {
        next_utf8_boundary(&self.draft, pos)
    }

    fn next_char_width(&self, pos: usize) -> usize {
        self.next_char_boundary(pos) - pos
    }

    fn delete_backward_word(&mut self) {
        let start = self.beginning_of_previous_word();
        if start == self.cursor {
            return;
        }
        let removed = self.draft[start..self.cursor].to_owned();
        self.set_kill_buffer(&removed);
        self.draft.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    fn kill_to_beginning_of_line(&mut self) {
        let start = self.current_line_start(self.cursor);
        if start == self.cursor {
            return;
        }
        let removed = self.draft[start..self.cursor].to_owned();
        self.set_kill_buffer(&removed);
        self.draft.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    fn kill_to_end_of_line(&mut self) {
        let end = self.current_line_end(self.cursor);
        if end == self.cursor {
            return;
        }
        let removed = self.draft[self.cursor..end].to_owned();
        self.set_kill_buffer(&removed);
        self.draft.replace_range(self.cursor..end, "");
    }

    fn begin_history_search(&mut self) -> Result<()> {
        if self.search.is_some() {
            self.search_backward();
            return Ok(());
        }
        if self.history.is_empty() {
            self.set_notice("history is empty");
            return Ok(());
        }

        self.search = Some(Search {
            query: String::new(),
            original_draft: self.draft.clone(),
            original_cursor: self.cursor,
            original_history_index: self.history_index,
            match_index: None,
        });
        Ok(())
    }

    fn search_backward(&mut self) {
        let next = if let Some(search) = &self.search {
            self.find_history_match_reverse(&search.query, search.match_index)
        } else {
            None
        };
        if let Some(search) = &mut self.search {
            search.match_index = next;
        }
        self.apply_search_preview();
    }

    fn search_forward(&mut self) {
        let next = if let Some(search) = &self.search {
            self.find_history_match_forward(&search.query, search.match_index)
        } else {
            None
        };
        if let Some(search) = &mut self.search {
            search.match_index = next;
        }
        self.apply_search_preview();
    }

    fn apply_search_preview(&mut self) {
        let Some(search) = &self.search else {
            return;
        };

        if let Some(index) = search.match_index {
            let text = self.history[index].clone();
            self.set_draft_text(&text);
            return;
        }

        let text = search.original_draft.clone();
        let cursor = min_usize(search.original_cursor, text.len());
        self.set_draft_text(&text);
        self.cursor = cursor;
    }

    fn update_search_preview(&mut self) {
        let next = match &self.search {
            Some(search) if search.query.is_empty() => None,
            Some(search) => self.find_history_match_reverse(&search.query, None),
            None => return,
        };

        if let Some(search) = &mut self.search {
            search.match_index = next;
        }
        self.apply_search_preview();
    }

    fn cancel_search(&mut self) {
        let Some(search) = self.search.take() else {
            return;
        };
        self.history_index = search.original_history_index;
        self.set_draft_text(&search.original_draft);
        self.cursor = min_usize(search.original_cursor, self.draft.len());
        self.set_notice("history search cancelled");
    }

    fn accept_search(&mut self) {
        let Some(search) = self.search.take() else {
            return;
        };

        if let Some(index) = search.match_index {
            if search.original_history_index.is_none() {
                self.history_draft = Some(search.original_draft);
            }
            self.history_index = Some(index);
            self.cursor = self.draft.len();
            self.set_notice("history match loaded into the draft");
            return;
        }

        self.history_index = search.original_history_index;
        self.set_draft_text(&search.original_draft);
        self.cursor = min_usize(search.original_cursor, self.draft.len());
        self.set_notice("history search had no match");
    }

    fn find_history_match_reverse(&self, query: &str, current: Option<usize>) -> Option<usize> {
        let mut index = current.unwrap_or(self.history.len());
        while index > 0 {
            index -= 1;
            if contains_ignore_case(&self.history[index], query) {
                return Some(index);
            }
        }
        None
    }

    fn find_history_match_forward(&self, query: &str, current: Option<usize>) -> Option<usize> {
        let mut index = current.map_or(0, |value| value + 1);
        while index < self.history.len() {
            if contains_ignore_case(&self.history[index], query) {
                return Some(index);
            }
            index += 1;
        }
        None
    }

    fn beginning_of_previous_word(&self) -> usize {
        let mut pos = self.cursor;
        while pos > 0 {
            let start = self.prev_char_boundary(pos);
            if !is_whitespace_byte(self.draft.as_bytes()[start]) {
                pos = start;
                break;
            }
            pos = start;
        }

        while pos > 0 {
            let prev = self.prev_char_boundary(pos);
            let current_class = byte_class(self.draft.as_bytes()[pos]);
            if byte_class(self.draft.as_bytes()[prev]) != current_class {
                break;
            }
            pos = prev;
        }
        pos
    }

    fn end_of_next_word(&self) -> usize {
        let mut pos = self.cursor;
        while pos < self.draft.len() && is_whitespace_byte(self.draft.as_bytes()[pos]) {
            pos = self.next_char_boundary(pos);
        }
        if pos >= self.draft.len() {
            return self.draft.len();
        }

        let class = byte_class(self.draft.as_bytes()[pos]);
        while pos < self.draft.len() && byte_class(self.draft.as_bytes()[pos]) == class {
            pos = self.next_char_boundary(pos);
        }
        pos
    }

    fn record_history(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if self.history.last().is_some_and(|last| last == text) {
            return;
        }
        self.history.push(text.to_owned());
    }

    fn handle_slash_command(
        &mut self,
        line: &str,
        requested_queue: bool,
    ) -> Result<CommandOutcome> {
        if !line.starts_with('/') {
            return Ok(CommandOutcome::NotCommand);
        }

        let body = trim_left_spaces(&line[1..]);
        if body.is_empty() {
            self.set_notice("commands: /help /clear /remix /theme /model");
            return Ok(CommandOutcome::Handled);
        }

        let split = body.find(char::is_whitespace);
        let (name, arg) = match split {
            Some(index) => (&body[..index], body[index + 1..].trim()),
            None => (body, ""),
        };

        if name.eq_ignore_ascii_case("help") {
            self.show_help_note();
            return Ok(CommandOutcome::Handled);
        }
        if name.eq_ignore_ascii_case("clear") {
            if self.active_request.is_some() || !self.queue.is_empty() {
                self.set_notice("wait for the active/queued turns before clearing");
                return Ok(CommandOutcome::Rejected);
            }
            self.clear_conversation();
            self.reset_history_navigation();
            self.set_notice("cleared transcript");
            return Ok(CommandOutcome::Handled);
        }
        if name.eq_ignore_ascii_case("remix") {
            self.remix_last_prompt(requested_queue)?;
            return Ok(CommandOutcome::Handled);
        }
        if name.eq_ignore_ascii_case("theme") {
            if arg.is_empty() {
                self.set_notice(format!(
                    "theme: {} | /theme next or /theme <name>",
                    THEMES[self.theme_index].name
                ));
                return Ok(CommandOutcome::Handled);
            }
            if arg.eq_ignore_ascii_case("next") {
                self.theme_index = (self.theme_index + 1) % THEMES.len();
                self.set_notice(format!(
                    "theme switched to {}",
                    THEMES[self.theme_index].name
                ));
                return Ok(CommandOutcome::Handled);
            }
            for (index, theme) in THEMES.iter().enumerate() {
                if arg.eq_ignore_ascii_case(theme.name) {
                    self.theme_index = index;
                    self.set_notice(format!("theme switched to {}", theme.name));
                    return Ok(CommandOutcome::Handled);
                }
            }
            self.set_notice(format!("unknown theme: {}", arg));
            return Ok(CommandOutcome::Rejected);
        }
        if name.eq_ignore_ascii_case("model") {
            return self.handle_model_command(arg);
        }

        self.set_notice(format!("unknown command: /{}. try /help", name));
        Ok(CommandOutcome::Rejected)
    }

    fn handle_model_command(&mut self, arg: &str) -> Result<CommandOutcome> {
        match parse_model_arg(arg) {
            ModelArg::Empty => {
                self.show_model_note();
                Ok(CommandOutcome::Handled)
            }
            ModelArg::Invalid => {
                self.set_notice("model must be a single token");
                Ok(CommandOutcome::Rejected)
            }
            ModelArg::Selected(model) => {
                self.current_model = model.clone();
                match self.save_current_model() {
                    Ok(()) => self.set_notice(format!("model switched to {}", model)),
                    Err(err) => self.set_notice(format!(
                        "model switched to {} (persistence failed: {})",
                        model, err
                    )),
                }
                Ok(CommandOutcome::Handled)
            }
        }
    }

    fn save_current_model(&self) -> Result<()> {
        let Some(path) = &self.config_path else {
            anyhow::bail!("no config path");
        };
        save_tui_config(path, &self.current_model)
    }

    fn show_model_note(&mut self) {
        self.append_note(&format!(
            "Model\n\
             current: {}\n\n\
             Aliases\n\
             {}\n\n\
             Custom single-token model IDs are passed through to the Responses API.",
            self.current_model,
            format_model_aliases()
        ));
    }

    fn show_help_note(&mut self) {
        self.append_note(
            "Shortcuts\n\
             Enter sends. Ctrl-J inserts a newline. Tab queues behind an active reply.\n\
             Up/Down move through wrapped draft lines and fall back to history at the edges. Ctrl-P and Ctrl-N walk history directly.\n\
             Ctrl-R searches history. Ctrl-C quits.\n\n\
             Commands\n\
             /help /clear /remix /theme [amp|oxide|ember|lagoon|next] /model [alias|id]",
        );
    }

    fn submit_draft(&mut self, requested_queue: bool) -> Result<()> {
        let trimmed = self
            .draft
            .trim_matches(|ch| matches!(ch, ' ' | '\n' | '\r' | '\t'))
            .to_owned();
        if trimmed.is_empty() {
            self.set_notice("draft is empty");
            return Ok(());
        }

        if trimmed == "?" {
            self.record_history(&trimmed);
            self.show_help_note();
            self.draft.clear();
            self.cursor = 0;
            self.reset_history_navigation();
            return Ok(());
        }

        match self.handle_slash_command(&trimmed, requested_queue)? {
            CommandOutcome::Handled => {
                self.record_history(&trimmed);
                self.draft.clear();
                self.cursor = 0;
                self.reset_history_navigation();
                return Ok(());
            }
            CommandOutcome::Rejected => return Ok(()),
            CommandOutcome::NotCommand => {}
        }

        let prompt = trimmed;
        self.record_history(&prompt);
        self.enqueue_prompt(prompt, requested_queue)?;
        self.scroll_from_bottom = 0;
        self.draft.clear();
        self.cursor = 0;
        self.reset_history_navigation();
        Ok(())
    }

    fn enqueue_prompt(&mut self, prompt: String, requested_queue: bool) -> Result<()> {
        self.messages.push(Message {
            role: Role::User,
            text: prompt.clone(),
        });
        self.messages.push(Message {
            role: Role::Agent,
            text: String::new(),
        });

        let job = Job {
            prompt,
            model: self.current_model.clone(),
            target_message: self.messages.len() - 1,
        };

        if self.active_request.is_some() && requested_queue {
            self.queue.push_back(job);
            self.set_notice("queued prompt");
            return Ok(());
        }

        if self.active_request.is_some() {
            self.queue.push_front(job);
            self.set_notice("scheduled next prompt");
            return Ok(());
        }

        self.start_request(job);
        self.clear_notice();
        Ok(())
    }

    fn remix_last_prompt(&mut self, requested_queue: bool) -> Result<()> {
        for index in (0..self.messages.len()).rev() {
            if self.messages[index].role == Role::User {
                let prompt = self.messages[index].text.clone();
                self.enqueue_prompt(prompt, requested_queue)?;
                return Ok(());
            }
        }

        self.set_notice("no user prompt yet");
        Ok(())
    }

    fn start_request(&mut self, job: Job) {
        let request_items = self.build_request_items(&job.prompt);
        let client = self.client.clone();
        let tx = self.worker_tx.clone();
        let worker_request = WorkerRequest {
            model: job.model.clone(),
            input_items: request_items,
        };

        thread::spawn(move || request_worker_main(client, tx, worker_request));

        self.active_request = Some(ActiveRequest {
            prompt: job.prompt,
            model: job.model,
            target_message: job.target_message,
        });
        self.spinner = self.spinner.wrapping_add(1);
    }

    fn build_request_items(&self, prompt: &str) -> Vec<InputItem> {
        let mut items = self
            .context
            .iter()
            .map(ContextItem::as_input_item)
            .collect::<Vec<_>>();
        items.push(InputItem::user_text(prompt.to_owned()));
        items
    }

    fn drain_worker_events(&mut self) -> Result<bool> {
        let mut dirty = false;
        let mut terminal = None::<bool>;

        loop {
            match self.worker_rx.try_recv() {
                Ok(event) => match event {
                    WorkerEvent::TextDelta(bytes) => {
                        if let Some(request) = &self.active_request {
                            self.messages[request.target_message].text.push_str(&bytes);
                        }
                        self.spinner = self.spinner.wrapping_add(1);
                        dirty = true;
                    }
                    WorkerEvent::Failed(message) => {
                        self.fail_active_request(&message);
                        terminal = Some(false);
                        dirty = true;
                    }
                    WorkerEvent::Completed(done) => {
                        self.complete_active_request(done);
                        terminal = Some(true);
                        dirty = true;
                    }
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        if let Some(completed) = terminal {
            if let Some(next) = self.queue.pop_front() {
                self.start_request(next);
                self.clear_notice();
            } else if completed {
                self.clear_notice();
            }
        }

        Ok(dirty)
    }

    fn complete_active_request(&mut self, done: TurnCompletion) {
        let Some(request) = self.active_request.take() else {
            return;
        };

        if self.messages[request.target_message].text.is_empty() && !done.output_text.is_empty() {
            self.messages[request.target_message]
                .text
                .push_str(&done.output_text);
        }

        self.context.push(ContextItem::UserText(request.prompt));
        for item in done.output_items {
            self.context.push(ContextItem::ResponseItem(item));
        }
    }

    fn fail_active_request(&mut self, message: &str) {
        let Some(request) = self.active_request.take() else {
            return;
        };

        self.set_notice(format!("request failed: {}", message));

        let target = request.target_message;
        if !self.messages[target].text.is_empty() {
            self.messages[target].text.push_str("\n\n");
        }
        self.messages[target].text.push_str("[request failed] ");
        self.messages[target].text.push_str(message);
    }

    fn clear_conversation(&mut self) {
        self.context.clear();
        self.messages.clear();
        self.scroll_from_bottom = 0;
    }

    fn set_notice(&mut self, notice: impl Into<String>) {
        self.notice = notice.into();
    }

    fn clear_notice(&mut self) {
        self.notice.clear();
    }

    fn render(&mut self) -> Result<()> {
        self.render_buf.clear();

        if self.size.cols < 48 || self.size.rows < 14 {
            self.render_too_small();
            return self.flush_render();
        }

        let theme = THEMES[self.theme_index];
        let box_margin = 1usize;
        let transcript_margin = 3usize;
        let box_outer_width = self.size.cols.saturating_sub(box_margin * 2);
        let compose_width = max_usize(box_outer_width.saturating_sub(4), 1);
        let transcript_width =
            max_usize(self.size.cols.saturating_sub(transcript_margin * 2 + 2), 1);

        let draft_lines = wrap_lines(&self.draft, compose_width);
        let draft_cursor = cursor_visual(&draft_lines, self.cursor);

        let info_height = 1usize;
        let max_compose_height = max_usize(self.size.rows.saturating_sub(info_height + 2), 1);
        let compose_height = compose_height_for(draft_lines.len(), max_compose_height);
        let compose_outer_height = compose_height + 2;
        let transcript_height = self
            .size
            .rows
            .saturating_sub(compose_outer_height + info_height);

        let transcript_lines = self.build_transcript_lines(transcript_width);
        let max_scroll = transcript_lines.len().saturating_sub(transcript_height);
        if self.scroll_from_bottom > max_scroll {
            self.scroll_from_bottom = max_scroll;
        }

        let transcript_end = transcript_lines
            .len()
            .saturating_sub(self.scroll_from_bottom);
        let transcript_start = transcript_end.saturating_sub(transcript_height);
        let transcript_visible = transcript_end.saturating_sub(transcript_start);
        let transcript_padding = transcript_height.saturating_sub(transcript_visible);

        let compose_view_end = max_usize(draft_cursor.line + 1, compose_height);
        let compose_visible_end = min_usize(compose_view_end, draft_lines.len());
        let compose_visible_start = compose_visible_end.saturating_sub(compose_height);

        self.render_buf.push_str("\x1b[?25l\x1b[H");
        let mut screen_row = 1usize;

        if self.messages.is_empty() {
            self.render_empty_splash(theme, transcript_height);
            screen_row += transcript_height;
        } else {
            for _ in 0..transcript_padding {
                self.new_line();
                screen_row += 1;
            }

            for line in &transcript_lines[transcript_start..transcript_end] {
                self.render_transcript_line(theme, line, transcript_margin, transcript_width);
                screen_row += 1;
            }
        }

        let top_label = if let Some(request) = &self.active_request {
            format!(
                "{} streaming {}",
                request.model,
                spinner_char(self.spinner) as char
            )
        } else {
            self.current_model.clone()
        };
        self.render_panel_border(
            theme,
            box_margin,
            box_outer_width,
            "╭",
            "╮",
            &top_label,
            theme.accent,
        );
        screen_row += 1;
        let compose_start_row = screen_row;

        for span in &draft_lines[compose_visible_start..compose_visible_end] {
            let text = self.draft[span.start..span.end].to_owned();
            self.render_panel_row(theme, box_margin, compose_width, &text);
            screen_row += 1;
        }

        for _ in 0..compose_height
            .saturating_sub(compose_visible_end.saturating_sub(compose_visible_start))
        {
            self.render_panel_row(theme, box_margin, compose_width, "");
            screen_row += 1;
        }

        let location_label = self.location_label.clone().unwrap_or_default();
        self.render_panel_border(
            theme,
            box_margin,
            box_outer_width,
            "╰",
            "╯",
            &location_label,
            theme.note,
        );
        let _ = screen_row;
        self.render_info_row(theme);

        let visible_cursor_line = draft_cursor.line.saturating_sub(compose_visible_start);
        let cursor_row = compose_start_row + visible_cursor_line;
        let cursor_col = box_margin + 3 + draft_cursor.col;
        let _ = write!(
            self.render_buf,
            "\x1b[{};{}H\x1b[?25h",
            cursor_row, cursor_col
        );

        self.flush_render()
    }

    fn render_too_small(&mut self) {
        self.render_buf.push_str("\x1b[?25l\x1b[H");
        self.render_buf.push_str("agentz\r\n");
        self.render_buf
            .push_str("Resize the terminal to at least 48x14.\r\n");
        self.render_buf.push_str("\x1b[3;1H\x1b[?25h");
    }

    fn render_empty_splash(&mut self, theme: Theme, height: usize) {
        let art = [
            "          .............",
            "      ....:::::::::::::....",
            "    ..::::--------------:::..",
            "   .::---====++++++++====---::.",
            "  .::--===+++++****+++++===--::.",
            " .::--==+++***######***+++==--::.",
            " .:--==++***##########***++==--:.",
            " .:--==++***##########***++==--:.",
            " .::--==+++***######***+++==--::.",
            "  .::--===+++++****+++++===--::.",
            "   .:::---====++++++====---:::.",
            "     ..::::------------::::..",
            "        ...::::::::::::...",
        ];
        let copy = [
            ("Welcome to Agentz", theme.accent, true),
            ("", theme.note, false),
            (
                "Streaming responses, local context, terminal-first.",
                theme.header,
                false,
            ),
            ("", theme.note, false),
            (
                "\"Build the loop first. Polish the agent second.\"",
                theme.note,
                false,
            ),
        ];
        let copy_offset = 3usize;
        let gap = 5usize;

        let art_width = art.iter().map(|line| line.len()).max().unwrap_or(0);
        let copy_width = copy.iter().map(|line| line.0.len()).max().unwrap_or(0);

        let block_height = max_usize(art.len(), copy_offset + copy.len());
        let block_width = art_width + gap + copy_width;
        if height == 0 {
            return;
        }

        if block_width + 4 > self.size.cols || block_height > height {
            let top_padding = (height.saturating_sub(2)) / 2;
            for _ in 0..top_padding {
                self.new_line();
            }
            let title = "Welcome to Agentz";
            let title_padding = (self.size.cols.saturating_sub(title.len())) / 2;
            self.append_repeat(" ", title_padding);
            self.set_color(theme.accent);
            self.render_buf.push_str("\x1b[1m");
            self.render_buf.push_str(title);
            self.reset_color();
            self.new_line();

            let subtitle = "Start typing below.";
            let subtitle_padding = (self.size.cols.saturating_sub(subtitle.len())) / 2;
            self.append_repeat(" ", subtitle_padding);
            self.set_color(theme.note);
            self.render_buf.push_str(subtitle);
            self.reset_color();
            self.new_line();

            for _ in 0..height.saturating_sub(top_padding + 2) {
                self.new_line();
            }
            return;
        }

        let top_padding = (height - block_height) / 2;
        let left_padding = (self.size.cols - block_width) / 2;

        for _ in 0..top_padding {
            self.new_line();
        }

        for block_row in 0..block_height {
            self.append_repeat(" ", left_padding);
            if let Some(line) = art.get(block_row) {
                self.set_color(theme.accent);
                self.render_buf.push_str(line);
                self.reset_color();
                self.append_repeat(" ", art_width.saturating_sub(line.len()));
            } else {
                self.append_repeat(" ", art_width);
            }

            self.append_repeat(" ", gap);
            if (copy_offset..copy_offset + copy.len()).contains(&block_row) {
                let (text, color, bold) = copy[block_row - copy_offset];
                self.set_color(color);
                if bold {
                    self.render_buf.push_str("\x1b[1m");
                }
                self.render_buf.push_str(text);
                self.reset_color();
            }
            self.new_line();
        }

        for _ in 0..height.saturating_sub(top_padding + block_height) {
            self.new_line();
        }
    }

    fn render_transcript_line(
        &mut self,
        theme: Theme,
        line: &TranscriptLine,
        margin: usize,
        width: usize,
    ) {
        let style = transcript_style(theme, line.role, line.first, line.active);

        self.append_repeat(" ", margin);
        if line.text.is_empty() {
            self.new_line();
            return;
        }

        self.set_color(style.color);
        if style.italic {
            self.render_buf.push_str("\x1b[3m");
        }
        if style.bold {
            self.render_buf.push_str("\x1b[1m");
        }
        self.render_buf.push_str(style.prefix);
        self.append_clipped(&line.text, width);
        self.reset_color();
        self.new_line();
    }

    fn render_panel_border(
        &mut self,
        theme: Theme,
        margin: usize,
        outer_width: usize,
        left: &str,
        right: &str,
        label: &str,
        label_color: u8,
    ) {
        let inner_width = outer_width.saturating_sub(2);
        let clipped_label = tail_clipped(label, inner_width);
        let fill_count = inner_width.saturating_sub(clipped_label.len());

        self.append_repeat(" ", margin);
        self.set_color(theme.note);
        self.render_buf.push_str(left);
        self.append_repeat("─", fill_count);
        if !clipped_label.is_empty() {
            self.set_color(label_color);
            self.render_buf.push_str(clipped_label);
            self.set_color(theme.note);
        }
        self.render_buf.push_str(right);
        self.reset_color();
        self.new_line();
    }

    fn render_panel_row(&mut self, theme: Theme, margin: usize, width: usize, text: &str) {
        let visible = clipped_prefix(text, width);
        let padding = width.saturating_sub(visible.len());

        self.append_repeat(" ", margin);
        self.set_color(theme.note);
        self.render_buf.push_str("│ ");
        self.reset_color();
        self.render_buf.push_str(visible);
        self.append_repeat(" ", padding);
        self.set_color(theme.note);
        self.render_buf.push_str(" │");
        self.reset_color();
        self.new_line();
    }

    fn render_info_row(&mut self, theme: Theme) {
        self.append_repeat(" ", 1);
        if let Some(search) = &self.search {
            let match_state = if search.query.is_empty() {
                "type to search history"
            } else if search.match_index.is_some() {
                "match"
            } else {
                "no match"
            };
            let line = format!("history search: {} [{}]", search.query, match_state);
            self.set_color(theme.note);
            self.append_clipped(&line, self.size.cols.saturating_sub(1));
            self.reset_color();
            self.clear_line();
            return;
        }

        let info = if self.notice.is_empty() {
            "? for shortcuts".to_owned()
        } else {
            self.notice.clone()
        };
        self.set_color(theme.note);
        self.append_clipped(&info, self.size.cols.saturating_sub(1));
        self.reset_color();
        self.clear_line();
    }

    fn build_transcript_lines(&self, width: usize) -> Vec<TranscriptLine> {
        let mut lines = Vec::new();
        if self.messages.is_empty() {
            return lines;
        }

        for (msg_index, message) in self.messages.iter().enumerate() {
            if msg_index != 0 {
                lines.push(TranscriptLine {
                    role: Role::Note,
                    text: String::new(),
                    first: false,
                    active: false,
                });
            }

            for (line_index, span) in wrap_lines(&message.text, width).into_iter().enumerate() {
                lines.push(TranscriptLine {
                    role: message.role,
                    text: message.text[span.start..span.end].to_owned(),
                    first: line_index == 0,
                    active: self
                        .active_request
                        .as_ref()
                        .is_some_and(|request| request.target_message == msg_index),
                });
            }
        }

        lines
    }

    fn set_color(&mut self, color: u8) {
        let _ = write!(self.render_buf, "\x1b[38;5;{}m", color);
    }

    fn reset_color(&mut self) {
        self.render_buf.push_str("\x1b[0m");
    }

    fn append_repeat(&mut self, text: &str, count: usize) {
        for _ in 0..count {
            self.render_buf.push_str(text);
        }
    }

    fn append_clipped(&mut self, text: &str, width: usize) {
        if width == 0 || text.is_empty() {
            return;
        }
        if text.len() <= width {
            self.render_buf.push_str(text);
            return;
        }
        if width <= 3 {
            self.render_buf.push_str(clipped_prefix(text, width));
            return;
        }

        self.render_buf.push_str(clipped_prefix(text, width - 3));
        self.render_buf.push_str("...");
    }

    fn new_line(&mut self) {
        self.render_buf.push_str("\x1b[K\r\n");
    }

    fn clear_line(&mut self) {
        self.render_buf.push_str("\x1b[K");
    }

    fn flush_render(&mut self) -> Result<()> {
        let mut stdout = io::stdout().lock();
        stdout
            .write_all(self.render_buf.as_bytes())
            .context("failed to write TUI frame")?;
        stdout.flush().context("failed to flush TUI frame")
    }
}

pub fn run(opts: Options) -> Result<()> {
    let _guard = TerminalGuard::init()?;
    let mut app = App::init(opts)?;
    app.run()
}

fn request_worker_main(client: openai::Client, tx: Sender<WorkerEvent>, request: WorkerRequest) {
    let result = (|| -> Result<()> {
        let delta_tx = tx.clone();
        let response = client.create_response_streaming(
            ResponsesRequest::init(request.model, Input::items(request.input_items))
                .with_store(false),
            move |delta| {
                let _ = delta_tx.send(WorkerEvent::TextDelta(delta.to_owned()));
                Ok(())
            },
            |_| Ok(()),
        )?;

        let output_text = response.output_text()?;
        let output_items = response.output_items()?;
        let _ = tx.send(WorkerEvent::Completed(TurnCompletion {
            output_text,
            output_items,
        }));
        Ok(())
    })();

    if let Err(err) = result {
        let _ = tx.send(WorkerEvent::Failed(err.to_string()));
    }
}

fn default_config_path() -> Option<PathBuf> {
    if let Some(dir) = non_empty_env("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(dir).join("agentz").join("config.json"));
    }
    non_empty_env("HOME").map(|home| {
        PathBuf::from(home)
            .join(".config")
            .join("agentz")
            .join("config.json")
    })
}

fn startup_model(config_path: Option<&Path>) -> String {
    config_path
        .and_then(load_config_model)
        .or_else(env_model)
        .unwrap_or_else(|| openai::DEFAULT_MODEL.to_owned())
}

fn load_config_model(path: &Path) -> Option<String> {
    let body = fs::read_to_string(path).ok()?;
    let config = serde_json::from_str::<TuiConfig>(&body).ok()?;
    normalize_model_value(&config.model)
}

fn save_tui_config(path: &Path, model: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("failed to create agentz config directory")?;
    }
    let mut body = serde_json::to_string_pretty(&TuiConfig {
        model: model.to_owned(),
    })
    .context("failed to serialize agentz config")?;
    body.push('\n');
    fs::write(path, body).context("failed to write agentz config")
}

fn env_model() -> Option<String> {
    non_empty_env("OPENAI_MODEL").and_then(|value| normalize_model_value(&value))
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn parse_model_arg(arg: &str) -> ModelArg {
    let mut words = arg.split_whitespace();
    let Some(first) = words.next() else {
        return ModelArg::Empty;
    };
    if words.next().is_some() {
        return ModelArg::Invalid;
    }
    normalize_model_value(first)
        .map(ModelArg::Selected)
        .unwrap_or(ModelArg::Invalid)
}

fn normalize_model_value(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.split_whitespace().nth(1).is_some() {
        return None;
    }
    Some(resolve_model_token(trimmed))
}

fn resolve_model_token(token: &str) -> String {
    MODEL_ALIASES
        .iter()
        .find(|alias| token.eq_ignore_ascii_case(alias.name))
        .map_or_else(|| token.to_owned(), |alias| alias.model.to_owned())
}

fn format_model_aliases() -> String {
    let mut lines = String::new();
    for (index, alias) in MODEL_ALIASES.iter().enumerate() {
        if index != 0 {
            lines.push('\n');
        }
        let _ = write!(lines, "{} -> {}", alias.name, alias.model);
    }
    lines
}

fn build_location_label() -> Result<Option<String>> {
    let pwd = current_pwd()?;
    let home = env::var("HOME").ok();
    let display_path = abbreviate_home(&pwd, home.as_deref());
    let branch = read_git_branch_name()?;

    Ok(Some(match branch {
        Some(name) => format!("{} ({})", display_path, name),
        None => display_path,
    }))
}

fn current_pwd() -> Result<String> {
    if let Ok(pwd) = env::var("PWD") {
        return Ok(pwd);
    }
    Ok(env::current_dir()
        .context("failed to get current directory")?
        .display()
        .to_string())
}

fn abbreviate_home(path: &str, home: Option<&str>) -> String {
    let Some(prefix) = home else {
        return path.to_owned();
    };
    if !path.starts_with(prefix) {
        return path.to_owned();
    }
    if path.len() != prefix.len() && path.as_bytes().get(prefix.len()) != Some(&b'/') {
        return path.to_owned();
    }
    if path.len() == prefix.len() {
        return "~".to_owned();
    }
    format!("~{}", &path[prefix.len()..])
}

fn read_git_branch_name() -> Result<Option<String>> {
    let head = match read_git_head()? {
        Some(head) => head,
        None => return Ok(None),
    };

    let trimmed = head.trim();
    if let Some(reference) = trimmed.strip_prefix("ref:") {
        let reference = reference.trim();
        let name = reference.rsplit('/').next().unwrap_or(reference);
        return Ok(Some(name.to_owned()));
    }

    let short_len = min_usize(trimmed.len(), 7);
    Ok(Some(trimmed[..short_len].to_owned()))
}

fn read_git_head() -> Result<Option<String>> {
    let dot_git = Path::new(".git");
    match fs::metadata(dot_git) {
        Ok(metadata) if metadata.is_dir() => {
            return Ok(Some(
                fs::read_to_string(dot_git.join("HEAD")).context("failed to read .git/HEAD")?,
            ));
        }
        Ok(_) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).context("failed to inspect .git"),
    }

    let dot_git_body = fs::read_to_string(dot_git).context("failed to read .git file")?;
    let trimmed = dot_git_body.trim();
    let Some(gitdir) = trimmed.strip_prefix("gitdir:") else {
        return Ok(None);
    };
    let gitdir = gitdir.trim();
    let head_path = resolve_gitdir(Path::new(gitdir));
    Ok(Some(
        fs::read_to_string(head_path.join("HEAD")).context("failed to read worktree HEAD")?,
    ))
}

fn resolve_gitdir(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        Path::new(".").join(path)
    }
}

fn read_terminal_size() -> Result<Size> {
    let (cols, rows) = terminal::size().context("failed to read terminal size")?;
    Ok(Size {
        cols: if cols == 0 { 80 } else { cols as usize },
        rows: if rows == 0 { 24 } else { rows as usize },
    })
}

fn wrap_lines(text: &str, width: usize) -> Vec<WrappedLine> {
    let mut lines = Vec::new();
    let effective_width = max_usize(width, 1);
    if text.is_empty() {
        lines.push(WrappedLine { start: 0, end: 0 });
        return lines;
    }

    let bytes = text.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        let start = index;
        let mut cursor = index;
        let mut count = 0usize;
        let mut last_space = None::<usize>;

        while cursor < bytes.len() {
            let byte = bytes[cursor];
            if byte == b'\n' {
                lines.push(WrappedLine { start, end: cursor });
                index = cursor + 1;
                if index == bytes.len() {
                    lines.push(WrappedLine {
                        start: index,
                        end: index,
                    });
                }
                break;
            }

            if byte == b' ' {
                last_space = Some(cursor);
            }
            count += 1;

            if count > effective_width {
                if let Some(space) = last_space {
                    if space == start {
                        lines.push(WrappedLine { start, end: cursor });
                        index = cursor;
                    } else {
                        lines.push(WrappedLine { start, end: space });
                        index = space + 1;
                    }
                } else {
                    lines.push(WrappedLine { start, end: cursor });
                    index = cursor;
                }
                break;
            }

            if cursor + 1 == bytes.len() {
                lines.push(WrappedLine {
                    start,
                    end: bytes.len(),
                });
                index = bytes.len();
                break;
            }

            cursor += 1;
        }
    }

    if lines.is_empty() {
        lines.push(WrappedLine { start: 0, end: 0 });
    }
    lines
}

fn cursor_visual(lines: &[WrappedLine], cursor: usize) -> CursorVisual {
    if lines.is_empty() {
        return CursorVisual { line: 0, col: 0 };
    }

    for (idx, line) in lines.iter().enumerate() {
        if cursor < line.start {
            return CursorVisual { line: idx, col: 0 };
        }
        if cursor <= line.end {
            return CursorVisual {
                line: idx,
                col: cursor - line.start,
            };
        }
    }

    let tail = lines[lines.len() - 1];
    CursorVisual {
        line: lines.len() - 1,
        col: tail.end - tail.start,
    }
}

struct CursorVisual {
    line: usize,
    col: usize,
}

fn key_from_event(event: KeyEvent) -> Option<Key> {
    let modifiers = event.modifiers;
    match event.code {
        KeyCode::Left => Some(Key::Left),
        KeyCode::Right => Some(Key::Right),
        KeyCode::Up => Some(Key::Up),
        KeyCode::Down => Some(Key::Down),
        KeyCode::Home => Some(Key::Home),
        KeyCode::End => Some(Key::End),
        KeyCode::Delete => Some(Key::Delete),
        KeyCode::Backspace => Some(Key::Backspace),
        KeyCode::PageUp => Some(Key::PageUp),
        KeyCode::PageDown => Some(Key::PageDown),
        KeyCode::Enter => Some(Key::Enter),
        KeyCode::Tab => Some(Key::Tab),
        KeyCode::Esc => Some(Key::Esc),
        KeyCode::Char(c) if modifiers.contains(KeyModifiers::CONTROL) => {
            match c.to_ascii_lowercase() {
                'a' => Some(Key::CtrlA),
                'b' => Some(Key::CtrlB),
                'c' => Some(Key::CtrlC),
                'd' => Some(Key::CtrlD),
                'e' => Some(Key::CtrlE),
                'f' => Some(Key::CtrlF),
                'j' => Some(Key::CtrlJ),
                'k' => Some(Key::CtrlK),
                'l' => Some(Key::CtrlL),
                'n' => Some(Key::CtrlN),
                'p' => Some(Key::CtrlP),
                'r' => Some(Key::CtrlR),
                's' => Some(Key::CtrlS),
                'u' => Some(Key::CtrlU),
                'w' => Some(Key::CtrlW),
                'y' => Some(Key::CtrlY),
                _ => None,
            }
        }
        KeyCode::Char(c) if modifiers.contains(KeyModifiers::ALT) => Some(Key::AltChar(c)),
        KeyCode::Char(c) => Some(Key::Char(c)),
        _ => None,
    }
}

fn spinner_char(index: usize) -> u8 {
    [b'|', b'/', b'-', b'\\'][index % 4]
}

fn clamp_usize(value: usize, low: usize, high: usize) -> usize {
    value.max(low).min(high)
}

fn min_usize(a: usize, b: usize) -> usize {
    a.min(b)
}

fn max_usize(a: usize, b: usize) -> usize {
    a.max(b)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ByteClass {
    Whitespace,
    Separator,
    Word,
}

fn prev_utf8_boundary(text: &str, pos: usize) -> usize {
    let mut i = min_usize(pos, text.len());
    if i == 0 {
        return 0;
    }
    i -= 1;
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn next_utf8_boundary(text: &str, pos: usize) -> usize {
    let mut i = min_usize(pos, text.len());
    if i >= text.len() {
        return text.len();
    }
    i += 1;
    while i < text.len() && !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

fn truncate_last_codepoint(buf: &mut String) {
    let next_len = prev_utf8_boundary(buf, buf.len());
    buf.truncate(next_len);
}

fn is_whitespace_byte(byte: u8) -> bool {
    matches!(byte, b' ' | b'\n' | b'\r' | b'\t')
}

fn is_word_separator(byte: u8) -> bool {
    matches!(
        byte,
        b'`' | b'~'
            | b'!'
            | b'@'
            | b'#'
            | b'$'
            | b'%'
            | b'^'
            | b'&'
            | b'*'
            | b'('
            | b')'
            | b'-'
            | b'='
            | b'+'
            | b'['
            | b'{'
            | b']'
            | b'}'
            | b'\\'
            | b'|'
            | b';'
            | b':'
            | b'\''
            | b'"'
            | b','
            | b'.'
            | b'<'
            | b'>'
            | b'/'
            | b'?'
    )
}

fn byte_class(byte: u8) -> ByteClass {
    if is_whitespace_byte(byte) {
        ByteClass::Whitespace
    } else if is_word_separator(byte) {
        ByteClass::Separator
    } else {
        ByteClass::Word
    }
}

fn ascii_lower(byte: u8) -> u8 {
    byte.to_ascii_lowercase()
}

fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    let haystack = haystack.as_bytes();
    let needle = needle.as_bytes();
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }

    for start in 0..=haystack.len() - needle.len() {
        let mut matched = true;
        for offset in 0..needle.len() {
            if ascii_lower(haystack[start + offset]) != ascii_lower(needle[offset]) {
                matched = false;
                break;
            }
        }
        if matched {
            return true;
        }
    }
    false
}

fn trim_left_spaces(text: &str) -> &str {
    text.trim_start_matches([' ', '\t'])
}

fn clipped_prefix(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn tail_clipped(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut start = text.len().saturating_sub(max_bytes);
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

fn compose_height_for(draft_line_count: usize, max_compose_height: usize) -> usize {
    min_usize(
        max_usize(draft_line_count, DEFAULT_COMPOSE_HEIGHT),
        max_compose_height,
    )
}

fn transcript_style(theme: Theme, role: Role, first: bool, active: bool) -> TranscriptStyle {
    match role {
        Role::User => TranscriptStyle {
            prefix: "| ",
            color: theme.user,
            italic: true,
            bold: false,
        },
        Role::Agent => TranscriptStyle {
            prefix: "  ",
            color: theme.agent,
            italic: false,
            bold: active && first,
        },
        Role::Note => TranscriptStyle {
            prefix: "  ",
            color: theme.note,
            italic: false,
            bold: false,
        },
    }
}

fn dirty_only() -> InputOutcome {
    InputOutcome {
        dirty: true,
        quit: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_config_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        env::temp_dir()
            .join(format!(
                "agentz-test-{}-{}-{}",
                std::process::id(),
                label,
                nanos
            ))
            .join("config.json")
    }

    fn test_app(config_path: Option<PathBuf>) -> App {
        let (worker_tx, worker_rx) = mpsc::channel();
        App {
            client: openai::Client::init("sk-test").expect("test client"),
            size: Size { cols: 80, rows: 24 },
            render_buf: String::new(),
            notice: String::new(),
            draft: String::new(),
            kill_buffer: String::new(),
            cursor: 0,
            scroll_from_bottom: 0,
            theme_index: 0,
            spinner: 0,
            location_label: None,
            current_model: openai::DEFAULT_MODEL.to_owned(),
            config_path,
            messages: Vec::new(),
            context: Vec::new(),
            queue: VecDeque::new(),
            history: Vec::new(),
            history_index: None,
            history_draft: None,
            search: None,
            worker_tx,
            worker_rx,
            active_request: None,
        }
    }

    #[test]
    fn wrap_lines_keeps_blank_rows_and_wraps_at_spaces() {
        let wrapped = wrap_lines("alpha beta\n\nomega", 5);
        assert_eq!(wrapped.len(), 4);
        assert_eq!(&"alpha beta"[wrapped[0].start..wrapped[0].end], "alpha");
        assert_eq!(wrapped[1].end, 10);
        assert_eq!(wrapped[2].start, 11);
        assert_eq!(wrapped[2].end, 11);
    }

    #[test]
    fn contains_ignore_case_matches_ascii_without_regard_to_case() {
        assert!(contains_ignore_case("Alpha Beta", "beta"));
        assert!(!contains_ignore_case("Alpha Beta", "gamma"));
    }

    #[test]
    fn transcript_style_renders_user_lines_as_blue_italic_pipe_blocks() {
        let style = transcript_style(THEMES[0], Role::User, true, false);
        assert_eq!(style.prefix, "| ");
        assert_eq!(style.color, 39);
        assert!(style.italic);
        assert!(!style.bold);
    }

    #[test]
    fn compose_height_uses_amp_sized_default_box() {
        assert_eq!(compose_height_for(1, 20), DEFAULT_COMPOSE_HEIGHT);
        assert_eq!(compose_height_for(6, 20), 6);
        assert_eq!(compose_height_for(1, 2), 2);
    }

    #[test]
    fn model_arg_resolves_aliases_defaults_and_custom_ids() {
        assert_eq!(
            parse_model_arg("latest"),
            ModelArg::Selected("gpt-5.5".to_owned())
        );
        assert_eq!(
            parse_model_arg("full"),
            ModelArg::Selected("gpt-5.4".to_owned())
        );
        assert_eq!(
            parse_model_arg("mini"),
            ModelArg::Selected("gpt-5.4-mini".to_owned())
        );
        assert_eq!(
            parse_model_arg("nano"),
            ModelArg::Selected("gpt-5.4-nano".to_owned())
        );
        assert_eq!(
            parse_model_arg("default"),
            ModelArg::Selected(openai::DEFAULT_MODEL.to_owned())
        );
        assert_eq!(
            parse_model_arg("gpt-custom-2026-04-24"),
            ModelArg::Selected("gpt-custom-2026-04-24".to_owned())
        );
    }

    #[test]
    fn model_arg_handles_empty_and_invalid_input() {
        assert_eq!(parse_model_arg(""), ModelArg::Empty);
        assert_eq!(parse_model_arg(" \t "), ModelArg::Empty);
        assert_eq!(parse_model_arg("mini extra"), ModelArg::Invalid);
    }

    #[test]
    fn config_load_save_round_trips_and_falls_back_for_bad_files() {
        let path = test_config_path("config");
        assert_eq!(load_config_model(&path), None);

        save_tui_config(&path, "gpt-5.4-mini").expect("save config");
        assert_eq!(load_config_model(&path), Some("gpt-5.4-mini".to_owned()));

        fs::write(&path, r#"{"model":"mini"}"#).expect("write alias config");
        assert_eq!(load_config_model(&path), Some("gpt-5.4-mini".to_owned()));

        fs::write(&path, "{").expect("write malformed config");
        assert_eq!(load_config_model(&path), None);

        fs::write(&path, r#"{"model":"bad model"}"#).expect("write invalid config");
        assert_eq!(load_config_model(&path), None);

        let _ = fs::remove_dir_all(path.parent().expect("test dir"));
    }

    #[test]
    fn model_command_without_args_lists_current_model_and_aliases() {
        let mut app = test_app(Some(test_config_path("model-list")));
        let outcome = app
            .handle_slash_command("/model", false)
            .expect("model command");

        assert_eq!(outcome, CommandOutcome::Handled);
        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].role, Role::Note);
        assert!(app.messages[0].text.contains(openai::DEFAULT_MODEL));
        assert!(app.messages[0].text.contains("latest -> gpt-5.5"));
        assert!(app.messages[0].text.contains("default -> "));
    }

    #[test]
    fn model_command_updates_current_model_and_persists() {
        let path = test_config_path("model-update");
        let mut app = test_app(Some(path.clone()));
        let outcome = app
            .handle_slash_command("/model mini", false)
            .expect("model command");

        assert_eq!(outcome, CommandOutcome::Handled);
        assert_eq!(app.current_model, "gpt-5.4-mini");
        assert!(app.notice.contains("model switched to gpt-5.4-mini"));
        assert_eq!(load_config_model(&path), Some("gpt-5.4-mini".to_owned()));

        let _ = fs::remove_dir_all(path.parent().expect("test dir"));
    }

    #[test]
    fn model_command_keeps_selection_when_persistence_fails() {
        let mut app = test_app(None);
        let outcome = app
            .handle_slash_command("/model nano", false)
            .expect("model command");

        assert_eq!(outcome, CommandOutcome::Handled);
        assert_eq!(app.current_model, "gpt-5.4-nano");
        assert!(app.notice.contains("persistence failed"));
    }

    #[test]
    fn queued_prompts_capture_the_selected_model_at_enqueue_time() {
        let mut app = test_app(Some(test_config_path("queue-model")));
        app.active_request = Some(ActiveRequest {
            prompt: "active".to_owned(),
            model: "gpt-active".to_owned(),
            target_message: 0,
        });
        app.current_model = "gpt-next".to_owned();

        app.enqueue_prompt("queued".to_owned(), true)
            .expect("enqueue prompt");

        assert_eq!(
            app.active_request.as_ref().expect("active request").model,
            "gpt-active"
        );
        assert_eq!(app.queue.len(), 1);
        assert_eq!(app.queue[0].model, "gpt-next");
    }
}
