const std = @import("std");
const openai = @import("agentz").openai;

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const File = std.Io.File;

pub const Options = struct {
    initial_prompt: ?[]const u8 = null,
};

const Theme = struct {
    name: []const u8,
    header: u8,
    accent: u8,
    user: u8,
    agent: u8,
    note: u8,
};

const themes = [_]Theme{
    .{ .name = "amp", .header = 255, .accent = 43, .user = 250, .agent = 255, .note = 245 },
    .{ .name = "oxide", .header = 39, .accent = 44, .user = 214, .agent = 81, .note = 244 },
    .{ .name = "ember", .header = 202, .accent = 208, .user = 220, .agent = 79, .note = 244 },
    .{ .name = "lagoon", .header = 45, .accent = 51, .user = 215, .agent = 86, .note = 244 },
};

const Role = enum {
    user,
    agent,
    note,
};

const Message = struct {
    role: Role,
    text: ArrayList(u8) = .empty,

    fn deinit(self: *Message, gpa: Allocator) void {
        self.text.deinit(gpa);
    }
};

const ContextItem = union(enum) {
    user_text: []u8,
    response_item_json: []u8,

    fn deinit(self: *ContextItem) void {
        switch (self.*) {
            .user_text => |bytes| std.heap.smp_allocator.free(bytes),
            .response_item_json => |bytes| std.heap.smp_allocator.free(bytes),
        }
        self.* = undefined;
    }

    fn asInputItem(self: ContextItem) openai.InputItem {
        return switch (self) {
            .user_text => |text| openai.InputItem.userText(text),
            .response_item_json => |json| openai.InputItem.rawJson(json),
        };
    }
};

const Job = struct {
    prompt: []u8,
    target_message: usize,

    fn deinit(self: *Job) void {
        std.heap.smp_allocator.free(self.prompt);
        self.* = undefined;
    }
};

const ActiveRequest = struct {
    prompt: []u8,
    request_items: []openai.InputItem,
    target_message: usize,

    fn deinit(self: *ActiveRequest) void {
        std.heap.smp_allocator.free(self.request_items);
        std.heap.smp_allocator.free(self.prompt);
        self.* = undefined;
    }
};

const TurnCompletion = struct {
    output_text: []u8,
    output_items: []openai.JsonBlob,

    fn deinit(self: *TurnCompletion) void {
        std.heap.smp_allocator.free(self.output_text);
        openai.Response.deinitJsonBlobs(std.heap.smp_allocator, self.output_items);
        self.* = undefined;
    }
};

const WorkerEvent = union(enum) {
    text_delta: []u8,
    failed: []u8,
    completed: TurnCompletion,

    fn deinit(self: *WorkerEvent) void {
        switch (self.*) {
            .text_delta => |bytes| std.heap.smp_allocator.free(bytes),
            .failed => |bytes| std.heap.smp_allocator.free(bytes),
            .completed => |*done| done.deinit(),
        }
        self.* = undefined;
    }
};

const WorkerQueue = struct {
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    events: ArrayList(WorkerEvent) = .empty,

    fn deinit(self: *WorkerQueue) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        for (self.events.items) |*event| event.deinit();
        self.events.deinit(std.heap.smp_allocator);
    }

    fn push(self: *WorkerQueue, event: WorkerEvent) !void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        try self.events.append(std.heap.smp_allocator, event);
    }

    fn takeAll(self: *WorkerQueue) ArrayList(WorkerEvent) {
        var out: ArrayList(WorkerEvent) = .empty;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        std.mem.swap(ArrayList(WorkerEvent), &out, &self.events);
        return out;
    }
};

const Search = struct {
    query: ArrayList(u8) = .empty,
    original_draft: []u8,
    original_cursor: usize,
    original_history_index: ?usize,
    match_index: ?usize = null,

    fn deinit(self: *Search, gpa: Allocator) void {
        self.query.deinit(gpa);
        gpa.free(self.original_draft);
    }
};

const WrappedLine = struct {
    start: usize,
    end: usize,
};

const TranscriptLine = struct {
    role: Role,
    slice: []const u8,
    first: bool,
    active: bool,
};

const InputOutcome = struct {
    dirty: bool = false,
    quit: bool = false,
};

const CommandOutcome = enum {
    not_command,
    handled,
    rejected,
};

const Key = union(enum) {
    char: u8,
    alt_char: u8,
    left,
    right,
    up,
    down,
    home,
    end,
    delete,
    backspace,
    page_up,
    page_down,
    enter,
    tab,
    esc,
    ctrl_a,
    ctrl_b,
    ctrl_c,
    ctrl_d,
    ctrl_e,
    ctrl_f,
    ctrl_j,
    ctrl_k,
    ctrl_l,
    ctrl_n,
    ctrl_p,
    ctrl_r,
    ctrl_s,
    ctrl_u,
    ctrl_w,
    ctrl_y,
};

const Size = struct {
    cols: usize,
    rows: usize,
};

const TerminalGuard = struct {
    io: std.Io,
    stdout: File,
    original: std.posix.termios,
    active: bool,

    fn init(io: std.Io) !TerminalGuard {
        const stdin = File.stdin();
        const stdout = File.stdout();
        if (!try stdin.isTty(io) or !try stdout.isTty(io)) return error.NotATerminal;
        try stdout.enableAnsiEscapeCodes(io);

        const original = try std.posix.tcgetattr(stdin.handle);
        var raw = original;
        raw.iflag.BRKINT = false;
        raw.iflag.ICRNL = false;
        raw.iflag.INPCK = false;
        raw.iflag.ISTRIP = false;
        raw.iflag.IXON = false;
        raw.lflag.ECHO = false;
        raw.lflag.ICANON = false;
        raw.lflag.IEXTEN = false;
        raw.lflag.ISIG = false;
        raw.oflag.OPOST = false;
        raw.cc[@intFromEnum(std.posix.V.MIN)] = 0;
        raw.cc[@intFromEnum(std.posix.V.TIME)] = 0;
        try std.posix.tcsetattr(stdin.handle, .FLUSH, raw);

        try File.writeStreamingAll(stdout, io, "\x1b[?1049h\x1b[2J\x1b[H\x1b[?25l");

        return .{
            .io = io,
            .stdout = stdout,
            .original = original,
            .active = true,
        };
    }

    fn deinit(self: *TerminalGuard) void {
        if (!self.active) return;
        _ = std.posix.tcsetattr(File.stdin().handle, .FLUSH, self.original) catch {};
        _ = File.writeStreamingAll(
            self.stdout,
            self.io,
            "\x1b[0m\x1b[2J\x1b[H\x1b[?25h\x1b[?1049l",
        ) catch {};
        self.active = false;
    }
};

const App = struct {
    gpa: Allocator,
    io: std.Io,
    client: openai.Client,
    stdout: File = .stdout(),
    size: Size,
    render_buf: ArrayList(u8) = .empty,
    input_buf: ArrayList(u8) = .empty,
    notice: ArrayList(u8) = .empty,
    draft: ArrayList(u8) = .empty,
    kill_buffer: ArrayList(u8) = .empty,
    cursor: usize = 0,
    scroll_from_bottom: usize = 0,
    theme_index: usize = 0,
    spinner: usize = 0,
    location_label: ?[]u8 = null,
    messages: ArrayList(Message) = .empty,
    context: ArrayList(ContextItem) = .empty,
    queue: ArrayList(Job) = .empty,
    history: ArrayList([]u8) = .empty,
    history_index: ?usize = null,
    history_draft: ?[]u8 = null,
    search: ?Search = null,
    worker_queue: WorkerQueue,
    request_thread: ?std.Thread = null,
    active_request: ?ActiveRequest = null,

    fn init(gpa: Allocator, io: std.Io, environ_map: *std.process.Environ.Map, opts: Options) !App {
        var app = App{
            .gpa = gpa,
            .io = io,
            .client = try openai.Client.fromEnv(std.heap.smp_allocator, io, environ_map),
            .size = try readTerminalSize(),
            .worker_queue = .{ .io = io },
        };
        app.location_label = try buildLocationLabel(gpa, io, environ_map);
        errdefer if (app.location_label) |label| gpa.free(label);
        if (opts.initial_prompt) |prompt| {
            try app.draft.appendSlice(gpa, prompt);
            app.cursor = app.draft.items.len;
        }
        return app;
    }

    fn deinit(self: *App) void {
        self.clearQueueAndRequest();
        if (self.active_request) |*request| request.deinit();
        self.worker_queue.deinit();
        for (self.context.items) |*item| item.deinit();
        self.context.deinit(self.gpa);
        self.client.deinit();
        for (self.messages.items) |*message| message.deinit(self.gpa);
        self.messages.deinit(self.gpa);
        self.draft.deinit(self.gpa);
        self.kill_buffer.deinit(self.gpa);
        self.notice.deinit(self.gpa);
        self.input_buf.deinit(self.gpa);
        self.render_buf.deinit(self.gpa);
        if (self.search) |*search| search.deinit(self.gpa);
        if (self.history_draft) |draft| self.gpa.free(draft);
        if (self.location_label) |label| self.gpa.free(label);
        for (self.history.items) |entry| self.gpa.free(entry);
        self.history.deinit(self.gpa);
    }

    fn run(self: *App) !void {
        try self.render();
        while (true) {
            if (try self.step()) break;
        }
    }

    fn step(self: *App) !bool {
        var dirty = try self.refreshSize();

        var fds = [_]std.posix.pollfd{.{
            .fd = File.stdin().handle,
            .events = std.posix.POLL.IN,
            .revents = 0,
        }};
        const timeout_ms: i32 = if (self.active_request == null) 80 else 16;
        const ready = try std.posix.poll(&fds, timeout_ms);

        if (fds[0].revents & (std.posix.POLL.IN | std.posix.POLL.HUP | std.posix.POLL.ERR) != 0) {
            const outcome = try self.readInput();
            dirty = dirty or outcome.dirty;
            if (outcome.quit) return true;
        } else if (ready == 0 and self.input_buf.items.len == 1 and self.input_buf.items[0] == 0x1b) {
            self.dropInput(1);
            const outcome = try self.handleKey(.esc);
            dirty = dirty or outcome.dirty;
            if (outcome.quit) return true;
        }

        dirty = (try self.drainWorkerEvents()) or dirty;
        if (dirty) try self.render();
        return false;
    }

    fn refreshSize(self: *App) !bool {
        const next = try readTerminalSize();
        if (next.cols == self.size.cols and next.rows == self.size.rows) return false;
        self.size = next;
        return true;
    }

    fn readInput(self: *App) !InputOutcome {
        var buf: [256]u8 = undefined;
        const read_n = try std.posix.read(File.stdin().handle, &buf);
        if (read_n == 0) return .{ .quit = true };

        try self.input_buf.appendSlice(self.gpa, buf[0..read_n]);
        var outcome = InputOutcome{};
        while (self.parseKey()) |key| {
            const next = try self.handleKey(key);
            outcome.dirty = outcome.dirty or next.dirty;
            outcome.quit = outcome.quit or next.quit;
            if (outcome.quit) break;
        }
        return outcome;
    }

    fn parseKey(self: *App) ?Key {
        const items = self.input_buf.items;
        if (items.len == 0) return null;

        if (items[0] == 0x1b) {
            if (items.len == 1) return null;
            if (items[1] == '[') {
                if (items.len < 3) return null;
                return switch (items[2]) {
                    'A' => self.consumeKey(3, .up),
                    'B' => self.consumeKey(3, .down),
                    'C' => self.consumeKey(3, .right),
                    'D' => self.consumeKey(3, .left),
                    'F' => self.consumeKey(3, .end),
                    'H' => self.consumeKey(3, .home),
                    'Z' => blk: {
                        self.dropInput(3);
                        break :blk null;
                    },
                    '3' => if (items.len >= 4 and items[3] == '~') self.consumeKey(4, .delete) else null,
                    '5' => if (items.len >= 4 and items[3] == '~') self.consumeKey(4, .page_up) else null,
                    '6' => if (items.len >= 4 and items[3] == '~') self.consumeKey(4, .page_down) else null,
                    '7' => if (items.len >= 4 and items[3] == '~') self.consumeKey(4, .home) else null,
                    '8' => if (items.len >= 4 and items[3] == '~') self.consumeKey(4, .end) else null,
                    else => blk: {
                        self.dropInput(1);
                        break :blk null;
                    },
                };
            }
            if (items[1] == 'O') {
                if (items.len < 3) return null;
                return switch (items[2]) {
                    'F' => self.consumeKey(3, .end),
                    'H' => self.consumeKey(3, .home),
                    else => blk: {
                        self.dropInput(1);
                        break :blk null;
                    },
                };
            }
            return self.consumeKey(2, .{ .alt_char = items[1] });
        }

        const byte = items[0];
        self.dropInput(1);
        return switch (byte) {
            1 => .ctrl_a,
            2 => .ctrl_b,
            3 => .ctrl_c,
            4 => .ctrl_d,
            5 => .ctrl_e,
            6 => .ctrl_f,
            9 => .tab,
            10 => .ctrl_j,
            11 => .ctrl_k,
            12 => .ctrl_l,
            14 => .ctrl_n,
            16 => .ctrl_p,
            13 => .enter,
            18 => .ctrl_r,
            19 => .ctrl_s,
            21 => .ctrl_u,
            23 => .ctrl_w,
            25 => .ctrl_y,
            27 => .esc,
            127, 8 => .backspace,
            else => .{ .char = byte },
        };
    }

    fn consumeKey(self: *App, count: usize, key: Key) Key {
        self.dropInput(count);
        return key;
    }

    fn dropInput(self: *App, count: usize) void {
        if (count >= self.input_buf.items.len) {
            self.input_buf.clearRetainingCapacity();
            return;
        }
        std.mem.copyForwards(
            u8,
            self.input_buf.items[0 .. self.input_buf.items.len - count],
            self.input_buf.items[count..],
        );
        self.input_buf.items.len -= count;
    }

    fn handleKey(self: *App, key: Key) !InputOutcome {
        if (self.search != null) return self.handleSearchKey(key);

        switch (key) {
            .ctrl_c => return .{ .quit = true },
            .esc => return .{},
            .ctrl_l => return .{ .dirty = true },
            .left => {
                self.moveCursorLeft();
                return .{ .dirty = true };
            },
            .right => {
                self.moveCursorRight();
                return .{ .dirty = true };
            },
            .ctrl_b => {
                self.moveCursorLeft();
                return .{ .dirty = true };
            },
            .ctrl_f => {
                self.moveCursorRight();
                return .{ .dirty = true };
            },
            .home, .ctrl_a => {
                self.cursor = self.currentLineStart(self.cursor);
                return .{ .dirty = true };
            },
            .end, .ctrl_e => {
                self.cursor = self.currentLineEnd(self.cursor);
                return .{ .dirty = true };
            },
            .up => {
                if (try self.moveCursorVertical(-1)) return .{ .dirty = true };
                if (try self.navigateHistoryOlder()) return .{ .dirty = true };
                return .{};
            },
            .down => {
                if (try self.moveCursorVertical(1)) return .{ .dirty = true };
                if (try self.navigateHistoryNewer()) return .{ .dirty = true };
                return .{};
            },
            .ctrl_p => {
                if (try self.navigateHistoryOlder()) return .{ .dirty = true };
                return .{};
            },
            .ctrl_n => {
                if (try self.navigateHistoryNewer()) return .{ .dirty = true };
                return .{};
            },
            .page_up => {
                self.scroll_from_bottom += clampUsize(self.size.rows / 2, 1, 12);
                return .{ .dirty = true };
            },
            .page_down => {
                self.scroll_from_bottom -|= clampUsize(self.size.rows / 2, 1, 12);
                return .{ .dirty = true };
            },
            .delete, .ctrl_d => {
                self.resetHistoryNavigation();
                if (self.cursor < self.draft.items.len) {
                    try self.draft.replaceRange(self.gpa, self.cursor, self.nextCharWidth(self.cursor), &.{});
                }
                return .{ .dirty = true };
            },
            .backspace => {
                self.resetHistoryNavigation();
                if (self.cursor > 0) {
                    const start = self.prevCharBoundary(self.cursor);
                    try self.draft.replaceRange(self.gpa, start, self.cursor - start, &.{});
                    self.cursor = start;
                }
                return .{ .dirty = true };
            },
            .ctrl_w => {
                self.resetHistoryNavigation();
                try self.deleteBackwardWord();
                return .{ .dirty = true };
            },
            .ctrl_u => {
                self.resetHistoryNavigation();
                try self.killToBeginningOfLine();
                return .{ .dirty = true };
            },
            .ctrl_k => {
                self.resetHistoryNavigation();
                try self.killToEndOfLine();
                return .{ .dirty = true };
            },
            .ctrl_y => {
                self.resetHistoryNavigation();
                try self.insertDraftSlice(self.kill_buffer.items);
                return .{ .dirty = true };
            },
            .ctrl_j => {
                self.resetHistoryNavigation();
                try self.insertDraftSlice("\n");
                return .{ .dirty = true };
            },
            .tab => {
                try self.submitDraft(true);
                return .{ .dirty = true };
            },
            .enter => {
                try self.submitDraft(false);
                return .{ .dirty = true };
            },
            .ctrl_r => {
                try self.beginHistorySearch();
                return .{ .dirty = true };
            },
            .ctrl_s => return .{},
            .alt_char => |byte| {
                switch (byte) {
                    'b', 'B' => self.cursor = self.beginningOfPreviousWord(),
                    'f', 'F' => self.cursor = self.endOfNextWord(),
                    else => {},
                }
                return .{ .dirty = true };
            },
            .char => |byte| {
                if (byte >= 32) {
                    self.resetHistoryNavigation();
                    try self.insertDraftSlice(&.{byte});
                }
                return .{ .dirty = true };
            },
        }
    }

    fn insertDraftSlice(self: *App, bytes: []const u8) !void {
        try self.draft.insertSlice(self.gpa, self.cursor, bytes);
        self.cursor += bytes.len;
    }

    fn setDraftText(self: *App, text: []const u8) !void {
        self.draft.clearRetainingCapacity();
        try self.draft.appendSlice(self.gpa, text);
        self.cursor = self.draft.items.len;
    }

    fn setKillBuffer(self: *App, text: []const u8) !void {
        self.kill_buffer.clearRetainingCapacity();
        try self.kill_buffer.appendSlice(self.gpa, text);
    }

    fn appendNote(self: *App, text: []const u8) !void {
        var note = Message{ .role = .note };
        errdefer note.deinit(self.gpa);
        try note.text.appendSlice(self.gpa, text);
        try self.messages.append(self.gpa, note);
        self.scroll_from_bottom = 0;
    }

    fn resetHistoryNavigation(self: *App) void {
        self.history_index = null;
        if (self.history_draft) |draft| self.gpa.free(draft);
        self.history_draft = null;
    }

    fn moveCursorLeft(self: *App) void {
        self.cursor = self.prevCharBoundary(self.cursor);
    }

    fn moveCursorRight(self: *App) void {
        self.cursor = self.nextCharBoundary(self.cursor);
    }

    fn moveCursorVertical(self: *App, delta: isize) !bool {
        const width = maxUsize(self.size.cols - 3, 1);
        var lines = try wrapLines(self.gpa, self.draft.items, width);
        defer lines.deinit(self.gpa);

        const pos = cursorVisual(lines.items, self.cursor);
        if (delta < 0 and pos.line == 0) return false;
        if (delta > 0 and pos.line + 1 >= lines.items.len) return false;

        const target_line = if (delta < 0) pos.line - 1 else pos.line + 1;
        const span = lines.items[target_line];
        self.cursor = span.start + minUsize(pos.col, span.end - span.start);
        return true;
    }

    fn currentLineStart(self: *App, pos: usize) usize {
        var i = minUsize(pos, self.draft.items.len);
        while (i > 0 and self.draft.items[i - 1] != '\n') : (i -= 1) {}
        return i;
    }

    fn currentLineEnd(self: *App, pos: usize) usize {
        var i = minUsize(pos, self.draft.items.len);
        while (i < self.draft.items.len and self.draft.items[i] != '\n') {
            i = self.nextCharBoundary(i);
        }
        return i;
    }

    fn navigateHistoryOlder(self: *App) !bool {
        if (self.history.items.len == 0) return false;

        if (self.history_index == null) {
            self.resetHistoryNavigation();
            self.history_draft = try self.gpa.dupe(u8, self.draft.items);
            self.history_index = self.history.items.len - 1;
        } else if (self.history_index.? == 0) {
            return false;
        } else {
            self.history_index = self.history_index.? - 1;
        }

        try self.setDraftText(self.history.items[self.history_index.?]);
        return true;
    }

    fn navigateHistoryNewer(self: *App) !bool {
        const index = self.history_index orelse return false;

        if (index + 1 < self.history.items.len) {
            self.history_index = index + 1;
            try self.setDraftText(self.history.items[self.history_index.?]);
            return true;
        }

        const draft = self.history_draft;
        self.history_index = null;
        self.history_draft = null;
        if (draft) |text| {
            defer self.gpa.free(text);
            try self.setDraftText(text);
        } else {
            self.draft.clearRetainingCapacity();
            self.cursor = 0;
        }
        return true;
    }

    fn prevCharBoundary(self: *App, pos: usize) usize {
        return prevUtf8Boundary(self.draft.items, pos);
    }

    fn nextCharBoundary(self: *App, pos: usize) usize {
        return nextUtf8Boundary(self.draft.items, pos);
    }

    fn nextCharWidth(self: *App, pos: usize) usize {
        return self.nextCharBoundary(pos) - pos;
    }

    fn deleteBackwardWord(self: *App) !void {
        const start = self.beginningOfPreviousWord();
        if (start == self.cursor) return;
        try self.setKillBuffer(self.draft.items[start..self.cursor]);
        try self.draft.replaceRange(self.gpa, start, self.cursor - start, &.{});
        self.cursor = start;
    }

    fn killToBeginningOfLine(self: *App) !void {
        const start = self.currentLineStart(self.cursor);
        if (start == self.cursor) return;
        try self.setKillBuffer(self.draft.items[start..self.cursor]);
        try self.draft.replaceRange(self.gpa, start, self.cursor - start, &.{});
        self.cursor = start;
    }

    fn killToEndOfLine(self: *App) !void {
        const end = self.currentLineEnd(self.cursor);
        if (end == self.cursor) return;
        try self.setKillBuffer(self.draft.items[self.cursor..end]);
        try self.draft.replaceRange(self.gpa, self.cursor, end - self.cursor, &.{});
    }

    fn beginHistorySearch(self: *App) !void {
        if (self.search != null) {
            try self.searchBackward();
            return;
        }
        if (self.history.items.len == 0) {
            try self.setNotice("history is empty", .{});
            return;
        }

        self.search = .{
            .query = .empty,
            .original_draft = try self.gpa.dupe(u8, self.draft.items),
            .original_cursor = self.cursor,
            .original_history_index = self.history_index,
            .match_index = null,
        };
    }

    fn searchBackward(self: *App) !void {
        if (self.search) |*search| {
            search.match_index = self.findHistoryMatchReverse(search.query.items, search.match_index);
            try self.applySearchPreview();
        }
    }

    fn searchForward(self: *App) !void {
        if (self.search) |*search| {
            search.match_index = self.findHistoryMatchForward(search.query.items, search.match_index);
            try self.applySearchPreview();
        }
    }

    fn applySearchPreview(self: *App) !void {
        const search = self.search orelse return;
        if (search.match_index) |index| {
            try self.setDraftText(self.history.items[index]);
            return;
        }

        try self.setDraftText(search.original_draft);
        self.cursor = minUsize(search.original_cursor, self.draft.items.len);
    }

    fn updateSearchPreview(self: *App) !void {
        if (self.search) |*search| {
            if (search.query.items.len == 0) {
                search.match_index = null;
                try self.applySearchPreview();
                return;
            }

            search.match_index = self.findHistoryMatchReverse(search.query.items, null);
            try self.applySearchPreview();
        }
    }

    fn handleSearchKey(self: *App, key: Key) !InputOutcome {
        switch (key) {
            .ctrl_c, .esc => {
                try self.cancelSearch();
                return .{ .dirty = true };
            },
            .enter => {
                try self.acceptSearch();
                return .{ .dirty = true };
            },
            .backspace => {
                if (self.search) |*search| {
                    truncateLastCodepoint(&search.query);
                    try self.updateSearchPreview();
                }
                return .{ .dirty = true };
            },
            .up, .ctrl_p, .ctrl_r => {
                try self.searchBackward();
                return .{ .dirty = true };
            },
            .down, .ctrl_n, .ctrl_s => {
                try self.searchForward();
                return .{ .dirty = true };
            },
            .char => |byte| {
                if (byte >= 32) {
                    if (self.search) |*search| {
                        try search.query.append(self.gpa, byte);
                        try self.updateSearchPreview();
                    }
                }
                return .{ .dirty = true };
            },
            else => return .{},
        }
    }

    fn cancelSearch(self: *App) !void {
        var search = self.search orelse return;
        defer search.deinit(self.gpa);
        self.search = null;
        self.history_index = search.original_history_index;
        try self.setDraftText(search.original_draft);
        self.cursor = minUsize(search.original_cursor, self.draft.items.len);
        try self.setNotice("history search cancelled", .{});
    }

    fn acceptSearch(self: *App) !void {
        var search = self.search orelse return;
        defer search.deinit(self.gpa);
        self.search = null;

        if (search.match_index) |index| {
            if (search.original_history_index == null) {
                if (self.history_draft) |draft| self.gpa.free(draft);
                self.history_draft = try self.gpa.dupe(u8, search.original_draft);
            }
            self.history_index = index;
            self.cursor = self.draft.items.len;
            try self.setNotice("history match loaded into the draft", .{});
            return;
        }

        self.history_index = search.original_history_index;
        try self.setDraftText(search.original_draft);
        self.cursor = minUsize(search.original_cursor, self.draft.items.len);
        try self.setNotice("history search had no match", .{});
    }

    fn findHistoryMatchReverse(self: *App, query: []const u8, current: ?usize) ?usize {
        var index = current orelse self.history.items.len;
        while (index > 0) {
            index -= 1;
            if (containsIgnoreCase(self.history.items[index], query)) return index;
        }
        return null;
    }

    fn findHistoryMatchForward(self: *App, query: []const u8, current: ?usize) ?usize {
        var index: usize = if (current) |value| value + 1 else 0;
        while (index < self.history.items.len) : (index += 1) {
            if (containsIgnoreCase(self.history.items[index], query)) return index;
        }
        return null;
    }

    fn beginningOfPreviousWord(self: *App) usize {
        var pos = self.cursor;
        while (pos > 0) {
            const start = self.prevCharBoundary(pos);
            if (!isWhitespaceByte(self.draft.items[start])) {
                pos = start;
                break;
            }
            pos = start;
        }
        while (pos > 0) {
            const prev = self.prevCharBoundary(pos);
            if (byteClass(self.draft.items[prev]) != byteClass(self.draft.items[pos])) break;
            pos = prev;
        }
        return pos;
    }

    fn endOfNextWord(self: *App) usize {
        var pos = self.cursor;
        while (pos < self.draft.items.len and isWhitespaceByte(self.draft.items[pos])) {
            pos = self.nextCharBoundary(pos);
        }
        if (pos >= self.draft.items.len) return self.draft.items.len;

        const class = byteClass(self.draft.items[pos]);
        while (pos < self.draft.items.len and byteClass(self.draft.items[pos]) == class) {
            pos = self.nextCharBoundary(pos);
        }
        return pos;
    }

    fn recordHistory(self: *App, text: []const u8) !void {
        if (text.len == 0) return;
        if (self.history.items.len != 0 and std.mem.eql(u8, self.history.items[self.history.items.len - 1], text)) {
            return;
        }
        try self.history.append(self.gpa, try self.gpa.dupe(u8, text));
    }

    fn handleSlashCommand(self: *App, line: []const u8, requested_queue: bool) !CommandOutcome {
        if (line.len == 0 or line[0] != '/') return .not_command;

        const body = trimLeftSpaces(line[1..]);
        if (body.len == 0) {
            try self.setNotice("commands: /help /clear /remix /theme", .{});
            return .handled;
        }

        const split = std.mem.indexOfAny(u8, body, " \t\r\n");
        const name = if (split) |i| body[0..i] else body;
        const arg = if (split) |i| std.mem.trim(u8, body[i + 1 ..], " \t\r\n") else "";

        if (std.ascii.eqlIgnoreCase(name, "help")) {
            try self.showHelpNote();
            return .handled;
        }
        if (std.ascii.eqlIgnoreCase(name, "clear")) {
            if (self.active_request != null or self.queue.items.len != 0) {
                try self.setNotice("wait for the active/queued turns before clearing", .{});
                return .rejected;
            }
            self.clearConversation();
            self.resetHistoryNavigation();
            try self.setNotice("cleared transcript", .{});
            return .handled;
        }
        if (std.ascii.eqlIgnoreCase(name, "remix")) {
            try self.remixLastPrompt(requested_queue);
            return .handled;
        }
        if (std.ascii.eqlIgnoreCase(name, "theme")) {
            if (arg.len == 0) {
                try self.setNotice(
                    "theme: {s} | /theme next or /theme <name>",
                    .{themes[self.theme_index].name},
                );
                return .handled;
            }
            if (std.ascii.eqlIgnoreCase(arg, "next")) {
                self.theme_index = (self.theme_index + 1) % themes.len;
                try self.setNotice("theme switched to {s}", .{themes[self.theme_index].name});
                return .handled;
            }
            for (themes, 0..) |theme, index| {
                if (std.ascii.eqlIgnoreCase(arg, theme.name)) {
                    self.theme_index = index;
                    try self.setNotice("theme switched to {s}", .{theme.name});
                    return .handled;
                }
            }
            try self.setNotice("unknown theme: {s}", .{arg});
            return .rejected;
        }

        try self.setNotice("unknown command: /{s}. try /help", .{name});
        return .rejected;
    }

    fn showHelpNote(self: *App) !void {
        try self.appendNote(
            "Shortcuts\n" ++
                "Enter sends. Ctrl-J inserts a newline. Tab queues behind an active reply.\n" ++
                "Up/Down move through wrapped draft lines and fall back to history at the edges. Ctrl-P and Ctrl-N walk history directly.\n" ++
                "Ctrl-R searches history. Ctrl-C quits.\n\n" ++
                "Commands\n" ++
                "/help /clear /remix /theme [amp|oxide|ember|lagoon|next]",
        );
    }

    fn submitDraft(self: *App, requested_queue: bool) !void {
        const trimmed = std.mem.trim(u8, self.draft.items, " \n\r\t");
        if (trimmed.len == 0) {
            try self.setNotice("draft is empty", .{});
            return;
        }

        if (std.mem.eql(u8, trimmed, "?")) {
            try self.recordHistory(trimmed);
            try self.showHelpNote();
            self.draft.clearRetainingCapacity();
            self.cursor = 0;
            self.resetHistoryNavigation();
            return;
        }

        switch (try self.handleSlashCommand(trimmed, requested_queue)) {
            .handled => {
                try self.recordHistory(trimmed);
                self.draft.clearRetainingCapacity();
                self.cursor = 0;
                self.resetHistoryNavigation();
                return;
            },
            .rejected => return,
            .not_command => {},
        }

        try self.recordHistory(trimmed);
        try self.enqueuePrompt(trimmed, requested_queue);
        self.scroll_from_bottom = 0;
        self.draft.clearRetainingCapacity();
        self.cursor = 0;
        self.resetHistoryNavigation();
    }

    fn enqueuePrompt(self: *App, prompt: []const u8, requested_queue: bool) !void {
        const prompt_copy = try std.heap.smp_allocator.dupe(u8, prompt);
        errdefer std.heap.smp_allocator.free(prompt_copy);

        var user_message = Message{ .role = .user };
        errdefer user_message.deinit(self.gpa);
        try user_message.text.appendSlice(self.gpa, prompt);
        try self.messages.append(self.gpa, user_message);

        var assistant_message = Message{ .role = .agent };
        errdefer assistant_message.deinit(self.gpa);
        try self.messages.append(self.gpa, assistant_message);

        const job = Job{
            .prompt = prompt_copy,
            .target_message = self.messages.items.len - 1,
        };
        if (self.active_request != null and requested_queue) {
            try self.queue.append(self.gpa, job);
            try self.setNotice("queued prompt", .{});
            return;
        }

        if (self.active_request != null) {
            try self.queue.insert(self.gpa, 0, job);
            try self.setNotice("scheduled next prompt", .{});
            return;
        }

        try self.startRequest(job);
        self.clearNotice();
    }

    fn remixLastPrompt(self: *App, requested_queue: bool) !void {
        var index = self.messages.items.len;
        while (index > 0) {
            index -= 1;
            const message = self.messages.items[index];
            if (message.role != .user) continue;

            try self.enqueuePrompt(message.text.items, requested_queue);
            return;
        }

        try self.setNotice("no user prompt yet", .{});
    }

    fn startRequest(self: *App, job: Job) !void {
        const prompt = job.prompt;
        errdefer std.heap.smp_allocator.free(prompt);

        const request_items = try self.buildRequestItems(prompt);
        errdefer std.heap.smp_allocator.free(request_items);

        const thread = try std.Thread.spawn(.{}, requestWorkerMain, .{WorkerArgs{
            .client = &self.client,
            .worker_queue = &self.worker_queue,
            .request_items = request_items,
        }});

        self.request_thread = thread;
        self.active_request = .{
            .prompt = prompt,
            .request_items = request_items,
            .target_message = job.target_message,
        };
        self.spinner +%= 1;
    }

    fn buildRequestItems(self: *App, prompt: []const u8) ![]openai.InputItem {
        const items = try std.heap.smp_allocator.alloc(openai.InputItem, self.context.items.len + 1);
        for (self.context.items, 0..) |item, index| {
            items[index] = item.asInputItem();
        }
        items[self.context.items.len] = openai.InputItem.userText(prompt);
        return items;
    }

    fn drainWorkerEvents(self: *App) !bool {
        var events = self.worker_queue.takeAll();
        defer events.deinit(std.heap.smp_allocator);
        if (events.items.len == 0) return false;

        var dirty = false;
        var terminal: enum { none, completed, failed } = .none;

        for (events.items) |*event| {
            switch (event.*) {
                .text_delta => |bytes| {
                    if (self.active_request) |request| {
                        try self.messages.items[request.target_message].text.appendSlice(self.gpa, bytes);
                    }
                    std.heap.smp_allocator.free(bytes);
                    self.spinner +%= 1;
                    dirty = true;
                },
                .failed => |message| {
                    try self.failActiveRequest(message);
                    std.heap.smp_allocator.free(message);
                    terminal = .failed;
                    dirty = true;
                },
                .completed => |*done| {
                    try self.completeActiveRequest(done);
                    terminal = .completed;
                    dirty = true;
                },
            }
        }

        if (terminal != .none) {
            self.finishRequestThread();
            if (self.queue.items.len != 0) {
                const next = self.queue.orderedRemove(0);
                try self.startRequest(next);
                self.clearNotice();
            } else if (terminal == .completed) {
                self.clearNotice();
            }
        }

        return dirty;
    }

    fn completeActiveRequest(self: *App, done: *TurnCompletion) !void {
        const request = self.active_request orelse {
            done.deinit();
            return;
        };
        self.active_request = null;

        if (self.messages.items[request.target_message].text.items.len == 0 and done.output_text.len != 0) {
            try self.messages.items[request.target_message].text.appendSlice(self.gpa, done.output_text);
        }
        std.heap.smp_allocator.free(done.output_text);

        try self.context.append(self.gpa, .{ .user_text = request.prompt });
        std.heap.smp_allocator.free(request.request_items);

        for (done.output_items) |item| {
            try self.context.append(self.gpa, .{ .response_item_json = item.bytes });
        }
        std.heap.smp_allocator.free(done.output_items);
    }

    fn failActiveRequest(self: *App, message: []const u8) !void {
        var request = self.active_request orelse return;
        self.active_request = null;
        try self.setNotice("request failed: {s}", .{message});

        const target = request.target_message;
        if (self.messages.items[target].text.items.len != 0) {
            try self.messages.items[target].text.appendSlice(self.gpa, "\n\n");
        }
        try self.messages.items[target].text.appendSlice(self.gpa, "[request failed] ");
        try self.messages.items[target].text.appendSlice(self.gpa, message);
        request.deinit();
    }

    fn finishRequestThread(self: *App) void {
        if (self.request_thread) |thread| {
            thread.join();
            self.request_thread = null;
        }
    }

    fn clearConversation(self: *App) void {
        for (self.context.items) |*item| item.deinit();
        self.context.clearRetainingCapacity();
        for (self.messages.items) |*message| message.deinit(self.gpa);
        self.messages.clearRetainingCapacity();
        self.scroll_from_bottom = 0;
    }

    fn clearQueueAndRequest(self: *App) void {
        self.finishRequestThread();
        var events = self.worker_queue.takeAll();
        defer events.deinit(std.heap.smp_allocator);
        for (events.items) |*event| event.deinit();
        for (self.queue.items) |*job| job.deinit();
        self.queue.clearRetainingCapacity();
    }

    fn setNotice(self: *App, comptime fmt: []const u8, args: anytype) !void {
        self.notice.clearRetainingCapacity();
        try self.notice.print(self.gpa, fmt, args);
    }

    fn clearNotice(self: *App) void {
        self.notice.clearRetainingCapacity();
    }

    fn render(self: *App) !void {
        self.render_buf.clearRetainingCapacity();

        if (self.size.cols < 48 or self.size.rows < 14) {
            try self.renderTooSmall();
            return;
        }

        const theme = themes[self.theme_index];
        const box_margin: usize = 1;
        const transcript_margin: usize = 3;
        const box_outer_width = self.size.cols -| (box_margin * 2);
        const compose_width = maxUsize(box_outer_width -| 4, 1);
        const transcript_width = maxUsize(self.size.cols -| (transcript_margin * 2 + 2), 1);

        var draft_lines = try wrapLines(self.gpa, self.draft.items, compose_width);
        defer draft_lines.deinit(self.gpa);
        const draft_cursor = cursorVisual(draft_lines.items, self.cursor);

        const info_height: usize = 1;
        const max_compose_height = maxUsize(self.size.rows -| (info_height + 2), 1);
        const compose_height = minUsize(maxUsize(draft_lines.items.len, 1), max_compose_height);
        const compose_outer_height = compose_height + 2;
        const transcript_height = self.size.rows -| (compose_outer_height + info_height);

        var transcript_lines = try self.buildTranscriptLines(transcript_width);
        defer transcript_lines.deinit(self.gpa);

        const max_scroll = if (transcript_lines.items.len > transcript_height)
            transcript_lines.items.len - transcript_height
        else
            0;
        if (self.scroll_from_bottom > max_scroll) self.scroll_from_bottom = max_scroll;

        const transcript_end = transcript_lines.items.len - self.scroll_from_bottom;
        const transcript_start = transcript_end -| transcript_height;
        const transcript_visible = transcript_end - transcript_start;
        const transcript_padding = transcript_height - transcript_visible;

        const compose_view_end = maxUsize(draft_cursor.line + 1, compose_height);
        const compose_visible_end = minUsize(compose_view_end, draft_lines.items.len);
        const compose_visible_start = compose_visible_end -| compose_height;

        try self.render_buf.appendSlice(self.gpa, "\x1b[?25l\x1b[H");
        var screen_row: usize = 1;

        if (self.messages.items.len == 0) {
            try self.renderEmptySplash(theme, transcript_height);
            screen_row += transcript_height;
        } else {
            var padding = transcript_padding;
            while (padding > 0) : (padding -= 1) {
                try self.newLine();
                screen_row += 1;
            }

            var line_index = transcript_start;
            while (line_index < transcript_end) : (line_index += 1) {
                try self.renderTranscriptLine(theme, transcript_lines.items[line_index], transcript_margin, transcript_width);
                screen_row += 1;
            }
        }

        var top_label_buf: [128]u8 = undefined;
        const top_label = if (self.active_request != null)
            try std.fmt.bufPrint(&top_label_buf, "{s} streaming {c}", .{ openai.DEFAULT_MODEL, spinnerChar(self.spinner) })
        else if (self.queue.items.len != 0)
            try std.fmt.bufPrint(&top_label_buf, "{s} {d} queued", .{ openai.DEFAULT_MODEL, self.queue.items.len })
        else
            openai.DEFAULT_MODEL;
        try self.renderPanelBorder(theme, box_margin, box_outer_width, "╭", "╮", top_label, theme.accent);
        screen_row += 1;
        const compose_start_row = screen_row;

        var compose_row = compose_visible_start;
        while (compose_row < compose_visible_end) : (compose_row += 1) {
            const span = draft_lines.items[compose_row];
            try self.renderPanelRow(theme, box_margin, compose_width, self.draft.items[span.start..span.end]);
            screen_row += 1;
        }

        var padding = compose_height - (compose_visible_end - compose_visible_start);
        while (padding > 0) : (padding -= 1) {
            try self.renderPanelRow(theme, box_margin, compose_width, "");
            screen_row += 1;
        }

        try self.renderPanelBorder(
            theme,
            box_margin,
            box_outer_width,
            "╰",
            "╯",
            self.location_label orelse "",
            theme.note,
        );
        screen_row += 1;

        try self.renderInfoRow(theme);

        const visible_cursor_line = draft_cursor.line -| compose_visible_start;
        const cursor_row = compose_start_row + visible_cursor_line;
        const cursor_col = box_margin + 3 + draft_cursor.col;
        try self.render_buf.print(self.gpa, "\x1b[{d};{d}H\x1b[?25h", .{ cursor_row, cursor_col });

        try File.writeStreamingAll(self.stdout, self.io, self.render_buf.items);
    }

    fn renderTooSmall(self: *App) !void {
        self.render_buf.clearRetainingCapacity();
        try self.render_buf.appendSlice(self.gpa, "\x1b[?25l\x1b[H");
        try self.render_buf.appendSlice(self.gpa, "agentz\r\n");
        try self.render_buf.appendSlice(self.gpa, "Resize the terminal to at least 48x14.\r\n");
        try self.render_buf.appendSlice(self.gpa, "\x1b[3;1H\x1b[?25h");
        try File.writeStreamingAll(self.stdout, self.io, self.render_buf.items);
    }

    fn renderEmptySplash(self: *App, theme: Theme, height: usize) !void {
        const art = [_][]const u8{
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
        };
        const copy = [_]struct { text: []const u8, color: u8, bold: bool }{
            .{ .text = "Welcome to Agentz", .color = theme.accent, .bold = true },
            .{ .text = "", .color = theme.note, .bold = false },
            .{ .text = "Streaming responses, local context, terminal-first.", .color = theme.header, .bold = false },
            .{ .text = "", .color = theme.note, .bold = false },
            .{ .text = "\"Build the loop first. Polish the agent second.\"", .color = theme.note, .bold = false },
        };
        const copy_offset: usize = 3;
        const gap: usize = 5;

        var art_width: usize = 0;
        for (art) |line| art_width = maxUsize(art_width, line.len);

        var copy_width: usize = 0;
        for (copy) |line| copy_width = maxUsize(copy_width, line.text.len);

        const block_height = maxUsize(art.len, copy_offset + copy.len);
        const block_width = art_width + gap + copy_width;
        if (height == 0) return;

        if (block_width + 4 > self.size.cols or block_height > height) {
            const top_padding = (height -| 2) / 2;
            var row: usize = 0;
            while (row < top_padding) : (row += 1) try self.newLine();
            const left_padding = (self.size.cols -| "Welcome to Agentz".len) / 2;
            try self.appendRepeat(" ", left_padding);
            try self.setColor(theme.accent);
            try self.render_buf.appendSlice(self.gpa, "\x1b[1mWelcome to Agentz");
            try self.resetColor();
            try self.newLine();
            const subtitle = "Start typing below.";
            const subtitle_padding = (self.size.cols -| subtitle.len) / 2;
            try self.appendRepeat(" ", subtitle_padding);
            try self.setColor(theme.note);
            try self.render_buf.appendSlice(self.gpa, subtitle);
            try self.resetColor();
            try self.newLine();
            row += 2;
            while (row < height) : (row += 1) try self.newLine();
            return;
        }

        const top_padding = (height - block_height) / 2;
        const left_padding = (self.size.cols - block_width) / 2;
        var row: usize = 0;
        while (row < top_padding) : (row += 1) try self.newLine();

        while (row < top_padding + block_height) : (row += 1) {
            const block_row = row - top_padding;
            try self.appendRepeat(" ", left_padding);

            if (block_row < art.len) {
                try self.setColor(theme.accent);
                try self.render_buf.appendSlice(self.gpa, art[block_row]);
                try self.resetColor();
                try self.appendRepeat(" ", art_width - art[block_row].len);
            } else {
                try self.appendRepeat(" ", art_width);
            }

            try self.appendRepeat(" ", gap);

            if (block_row >= copy_offset and block_row < copy_offset + copy.len) {
                const line = copy[block_row - copy_offset];
                try self.setColor(line.color);
                if (line.bold) try self.render_buf.appendSlice(self.gpa, "\x1b[1m");
                try self.render_buf.appendSlice(self.gpa, line.text);
                try self.resetColor();
            }

            try self.newLine();
        }

        while (row < height) : (row += 1) try self.newLine();
    }

    fn renderTranscriptLine(
        self: *App,
        theme: Theme,
        line: TranscriptLine,
        margin: usize,
        width: usize,
    ) !void {
        const prefix = switch (line.role) {
            .user => if (line.first) "› " else "  ",
            .agent => "  ",
            .note => "  ",
        };
        const color = switch (line.role) {
            .user => theme.user,
            .agent => theme.agent,
            .note => theme.note,
        };

        try self.appendRepeat(" ", margin);
        if (line.slice.len == 0) {
            try self.newLine();
            return;
        }

        try self.setColor(theme.note);
        try self.render_buf.appendSlice(self.gpa, prefix);
        try self.resetColor();
        try self.setColor(color);
        if (line.active and line.first) try self.render_buf.appendSlice(self.gpa, "\x1b[1m");
        try self.appendClipped(line.slice, width);
        try self.resetColor();
        try self.newLine();
    }

    fn renderPanelBorder(
        self: *App,
        theme: Theme,
        margin: usize,
        outer_width: usize,
        left: []const u8,
        right: []const u8,
        label: []const u8,
        label_color: u8,
    ) !void {
        const inner_width = outer_width -| 2;
        const clipped_label = if (label.len > inner_width) label[label.len - inner_width ..] else label;
        const fill_count = inner_width - clipped_label.len;

        try self.appendRepeat(" ", margin);
        try self.setColor(theme.note);
        try self.render_buf.appendSlice(self.gpa, left);
        try self.appendRepeat("─", fill_count);
        if (clipped_label.len != 0) {
            try self.setColor(label_color);
            try self.render_buf.appendSlice(self.gpa, clipped_label);
            try self.setColor(theme.note);
        }
        try self.render_buf.appendSlice(self.gpa, right);
        try self.resetColor();
        try self.newLine();
    }

    fn renderPanelRow(self: *App, theme: Theme, margin: usize, width: usize, text: []const u8) !void {
        const visible_len = minUsize(text.len, width);
        const padding = width - visible_len;

        try self.appendRepeat(" ", margin);
        try self.setColor(theme.note);
        try self.render_buf.appendSlice(self.gpa, "│ ");
        try self.resetColor();
        if (visible_len != 0) try self.appendClipped(text, width);
        try self.appendRepeat(" ", padding);
        try self.setColor(theme.note);
        try self.render_buf.appendSlice(self.gpa, " │");
        try self.resetColor();
        try self.newLine();
    }

    fn renderInfoRow(self: *App, theme: Theme) !void {
        try self.appendRepeat(" ", 1);
        if (self.search) |search| {
            const match_state = if (search.query.items.len == 0)
                "type to search history"
            else if (search.match_index != null)
                "match"
            else
                "no match";
            var footer: [512]u8 = undefined;
            const line = try std.fmt.bufPrint(
                &footer,
                "history search: {s} [{s}]",
                .{ search.query.items, match_state },
            );
            try self.setColor(theme.note);
            try self.appendClipped(line, self.size.cols -| 1);
            try self.resetColor();
            try self.clearLine();
            return;
        }

        const info = if (self.notice.items.len != 0) self.notice.items else "? for shortcuts";
        try self.setColor(theme.note);
        try self.appendClipped(info, self.size.cols -| 1);
        try self.resetColor();
        try self.clearLine();
    }

    fn buildTranscriptLines(self: *App, width: usize) !ArrayList(TranscriptLine) {
        var lines: ArrayList(TranscriptLine) = .empty;
        errdefer lines.deinit(self.gpa);

        if (self.messages.items.len == 0) {
            return lines;
        }

        for (self.messages.items, 0..) |message, msg_index| {
            if (msg_index != 0) {
                try lines.append(self.gpa, .{
                    .role = .note,
                    .slice = "",
                    .first = false,
                    .active = false,
                });
            }
            var wrapped = try wrapLines(self.gpa, message.text.items, width);
            defer wrapped.deinit(self.gpa);

            for (wrapped.items, 0..) |span, line_index| {
                try lines.append(self.gpa, .{
                    .role = message.role,
                    .slice = message.text.items[span.start..span.end],
                    .first = line_index == 0,
                    .active = if (self.active_request) |request| request.target_message == msg_index else false,
                });
            }
        }

        return lines;
    }

    fn setColor(self: *App, color: u8) !void {
        try self.render_buf.print(self.gpa, "\x1b[38;5;{d}m", .{color});
    }

    fn resetColor(self: *App) !void {
        try self.render_buf.appendSlice(self.gpa, "\x1b[0m");
    }

    fn appendRepeat(self: *App, text: []const u8, count: usize) !void {
        var i: usize = 0;
        while (i < count) : (i += 1) {
            try self.render_buf.appendSlice(self.gpa, text);
        }
    }

    fn appendClipped(self: *App, text: []const u8, width: usize) !void {
        if (width == 0 or text.len == 0) return;
        if (text.len <= width) {
            try self.render_buf.appendSlice(self.gpa, text);
            return;
        }

        if (width <= 3) {
            try self.render_buf.appendSlice(self.gpa, text[0..width]);
            return;
        }

        try self.render_buf.appendSlice(self.gpa, text[0 .. width - 3]);
        try self.render_buf.appendSlice(self.gpa, "...");
    }

    fn newLine(self: *App) !void {
        try self.render_buf.appendSlice(self.gpa, "\x1b[K\r\n");
    }

    fn clearLine(self: *App) !void {
        try self.render_buf.appendSlice(self.gpa, "\x1b[K");
    }
};

const WorkerArgs = struct {
    client: *openai.Client,
    worker_queue: *WorkerQueue,
    request_items: []const openai.InputItem,
};

pub fn run(init: std.process.Init, opts: Options) !void {
    var guard = try TerminalGuard.init(init.io);
    defer guard.deinit();

    var app = try App.init(init.gpa, init.io, init.environ_map, opts);
    defer app.deinit();
    try app.run();
}

fn requestWorkerMain(args: WorkerArgs) void {
    requestWorkerMainInner(args) catch |err| {
        const message = std.heap.smp_allocator.dupe(u8, @errorName(err)) catch return;
        args.worker_queue.push(.{ .failed = message }) catch std.heap.smp_allocator.free(message);
    };
}

fn requestWorkerMainInner(args: WorkerArgs) !void {
    var response = try args.client.*.createResponseStreaming(
        std.heap.smp_allocator,
        openai.ResponsesRequest.init(openai.DEFAULT_MODEL, .{ .items = args.request_items }).withStore(false),
        .{
            .context = args.worker_queue,
            .on_output_text_delta = onStreamTextDelta,
        },
    );
    defer response.deinit();

    const output_text = try response.outputTextAlloc(std.heap.smp_allocator);
    errdefer std.heap.smp_allocator.free(output_text);
    const output_items = try response.outputItemJsonBlobsAlloc(std.heap.smp_allocator);
    errdefer openai.Response.deinitJsonBlobs(std.heap.smp_allocator, output_items);

    try args.worker_queue.push(.{
        .completed = .{
            .output_text = output_text,
            .output_items = output_items,
        },
    });
}

fn onStreamTextDelta(context: ?*anyopaque, delta: []const u8) anyerror!void {
    const worker_queue: *WorkerQueue = @ptrCast(@alignCast(context orelse return));
    const text = try std.heap.smp_allocator.dupe(u8, delta);
    errdefer std.heap.smp_allocator.free(text);
    try worker_queue.push(.{ .text_delta = text });
}

fn buildLocationLabel(
    gpa: Allocator,
    io: std.Io,
    environ_map: *const std.process.Environ.Map,
) !?[]u8 {
    const pwd = environ_map.get("PWD") orelse return null;
    const home = environ_map.get("HOME");
    const display_path = try abbreviateHomeAlloc(gpa, pwd, home);
    errdefer gpa.free(display_path);
    const branch = try readGitBranchName(gpa, io);
    defer if (branch) |name| gpa.free(name);

    if (branch) |name| {
        defer gpa.free(display_path);
        return @as(?[]u8, try std.fmt.allocPrint(gpa, "{s} ({s})", .{ display_path, name }));
    }
    return @as(?[]u8, display_path);
}

fn abbreviateHomeAlloc(gpa: Allocator, path: []const u8, home: ?[]const u8) ![]u8 {
    const prefix = home orelse return gpa.dupe(u8, path);
    if (!std.mem.startsWith(u8, path, prefix)) return gpa.dupe(u8, path);
    if (path.len != prefix.len and path[prefix.len] != '/') return gpa.dupe(u8, path);
    if (path.len == prefix.len) return gpa.dupe(u8, "~");
    return std.fmt.allocPrint(gpa, "~{s}", .{path[prefix.len..]});
}

fn readGitBranchName(gpa: Allocator, io: std.Io) !?[]u8 {
    const head = readGitHeadAlloc(gpa, io) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer gpa.free(head);

    const trimmed = std.mem.trim(u8, head, " \r\n\t");
    if (std.mem.startsWith(u8, trimmed, "ref:")) {
        const ref = std.mem.trim(u8, trimmed[4..], " \t");
        const name = if (std.mem.lastIndexOfScalar(u8, ref, '/')) |idx| ref[idx + 1 ..] else ref;
        return @as(?[]u8, try gpa.dupe(u8, name));
    }

    const short_len = minUsize(trimmed.len, 7);
    return @as(?[]u8, try gpa.dupe(u8, trimmed[0..short_len]));
}

fn readGitHeadAlloc(gpa: Allocator, io: std.Io) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, ".git/HEAD", gpa, .limited(4096)) catch |err| switch (err) {
        error.FileNotFound => blk: {
            const dot_git = try std.Io.Dir.cwd().readFileAlloc(io, ".git", gpa, .limited(4096));
            defer gpa.free(dot_git);
            const trimmed = std.mem.trim(u8, dot_git, " \r\n\t");
            if (!std.mem.startsWith(u8, trimmed, "gitdir:")) return error.FileNotFound;
            const gitdir = std.mem.trim(u8, trimmed[7..], " \t");
            const head_path = try std.fmt.allocPrint(gpa, "{s}/HEAD", .{gitdir});
            defer gpa.free(head_path);
            break :blk try std.Io.Dir.cwd().readFileAlloc(io, head_path, gpa, .limited(4096));
        },
        else => return err,
    };
}

fn readTerminalSize() !Size {
    var winsize: std.posix.winsize = .{
        .row = 0,
        .col = 0,
        .xpixel = 0,
        .ypixel = 0,
    };
    const rc = std.os.linux.ioctl(
        File.stdout().handle,
        std.os.linux.T.IOCGWINSZ,
        @intFromPtr(&winsize),
    );
    return switch (std.posix.errno(rc)) {
        .SUCCESS => .{
            .cols = if (winsize.col == 0) 80 else winsize.col,
            .rows = if (winsize.row == 0) 24 else winsize.row,
        },
        else => .{ .cols = 80, .rows = 24 },
    };
}

fn wrapLines(gpa: Allocator, text: []const u8, width: usize) !ArrayList(WrappedLine) {
    var lines: ArrayList(WrappedLine) = .empty;
    errdefer lines.deinit(gpa);

    const effective_width = maxUsize(width, 1);
    if (text.len == 0) {
        try lines.append(gpa, .{ .start = 0, .end = 0 });
        return lines;
    }

    var index: usize = 0;
    while (index < text.len) {
        const start = index;
        var cursor = index;
        var count: usize = 0;
        var last_space: ?usize = null;

        while (cursor < text.len) : (cursor += 1) {
            const byte = text[cursor];
            if (byte == '\n') {
                try lines.append(gpa, .{ .start = start, .end = cursor });
                index = cursor + 1;
                if (index == text.len) {
                    try lines.append(gpa, .{ .start = index, .end = index });
                }
                break;
            }

            if (byte == ' ') last_space = cursor;
            count += 1;

            if (count > effective_width) {
                if (last_space) |space| {
                    if (space == start) {
                        try lines.append(gpa, .{ .start = start, .end = cursor });
                        index = cursor;
                    } else {
                        try lines.append(gpa, .{ .start = start, .end = space });
                        index = space + 1;
                    }
                } else {
                    try lines.append(gpa, .{ .start = start, .end = cursor });
                    index = cursor;
                }
                break;
            }

            if (cursor + 1 == text.len) {
                try lines.append(gpa, .{ .start = start, .end = text.len });
                index = text.len;
                break;
            }
        }
    }

    if (lines.items.len == 0) try lines.append(gpa, .{ .start = 0, .end = 0 });
    return lines;
}

fn cursorVisual(lines: []const WrappedLine, cursor: usize) struct { line: usize, col: usize } {
    if (lines.len == 0) return .{ .line = 0, .col = 0 };

    for (lines, 0..) |line, idx| {
        if (cursor < line.start) return .{ .line = idx, .col = 0 };
        if (cursor <= line.end) return .{ .line = idx, .col = cursor - line.start };
    }

    const tail = lines[lines.len - 1];
    return .{ .line = lines.len - 1, .col = tail.end - tail.start };
}

fn spinnerChar(index: usize) u8 {
    const chars = [_]u8{ '|', '/', '-', '\\' };
    return chars[index % chars.len];
}

fn clampUsize(value: usize, low: usize, high: usize) usize {
    return @min(@max(value, low), high);
}

fn minUsize(a: usize, b: usize) usize {
    return if (a < b) a else b;
}

fn maxUsize(a: usize, b: usize) usize {
    return if (a > b) a else b;
}

const ByteClass = enum {
    whitespace,
    separator,
    word,
};

fn prevUtf8Boundary(text: []const u8, pos: usize) usize {
    var i = minUsize(pos, text.len);
    if (i == 0) return 0;
    i -= 1;
    while (i > 0 and isUtf8Continuation(text[i])) : (i -= 1) {}
    return i;
}

fn nextUtf8Boundary(text: []const u8, pos: usize) usize {
    var i = minUsize(pos, text.len);
    if (i >= text.len) return text.len;
    i += 1;
    while (i < text.len and isUtf8Continuation(text[i])) : (i += 1) {}
    return i;
}

fn truncateLastCodepoint(buf: *ArrayList(u8)) void {
    buf.items.len = prevUtf8Boundary(buf.items, buf.items.len);
}

fn isUtf8Continuation(byte: u8) bool {
    return (byte & 0b1100_0000) == 0b1000_0000;
}

fn isWhitespaceByte(byte: u8) bool {
    return switch (byte) {
        ' ', '\n', '\r', '\t' => true,
        else => false,
    };
}

fn isWordSeparator(byte: u8) bool {
    return switch (byte) {
        '`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '=', '+', '[', '{', ']', '}', '\\', '|', ';', ':', '\'', '"', ',', '.', '<', '>', '/', '?' => true,
        else => false,
    };
}

fn byteClass(byte: u8) ByteClass {
    if (isWhitespaceByte(byte)) return .whitespace;
    if (isWordSeparator(byte)) return .separator;
    return .word;
}

fn asciiLower(byte: u8) u8 {
    return std.ascii.toLower(byte);
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;

    var start: usize = 0;
    while (start + needle.len <= haystack.len) : (start += 1) {
        var matched = true;
        var offset: usize = 0;
        while (offset < needle.len) : (offset += 1) {
            if (asciiLower(haystack[start + offset]) != asciiLower(needle[offset])) {
                matched = false;
                break;
            }
        }
        if (matched) return true;
    }
    return false;
}

fn trimLeftSpaces(text: []const u8) []const u8 {
    var start: usize = 0;
    while (start < text.len and (text[start] == ' ' or text[start] == '\t')) : (start += 1) {}
    return text[start..];
}

test "wrapLines keeps blank rows and wraps at spaces" {
    var wrapped = try wrapLines(std.testing.allocator, "alpha beta\n\nomega", 5);
    defer wrapped.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 4), wrapped.items.len);
    try std.testing.expectEqualStrings("alpha", "alpha beta"[wrapped.items[0].start..wrapped.items[0].end]);
    try std.testing.expectEqual(@as(usize, 10), wrapped.items[1].end);
    try std.testing.expectEqual(@as(usize, 11), wrapped.items[2].start);
    try std.testing.expectEqual(@as(usize, 11), wrapped.items[2].end);
}

test "containsIgnoreCase matches ASCII without regard to case" {
    try std.testing.expect(containsIgnoreCase("Alpha Beta", "beta"));
    try std.testing.expect(!containsIgnoreCase("Alpha Beta", "gamma"));
}
