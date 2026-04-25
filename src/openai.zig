const std = @import("std");

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const EnvironMap = std.process.Environ.Map;
const Io = std.Io;

const whitespace = &std.ascii.whitespace;
const dotenv_path = ".env.local";
const dotenv_max_bytes = 64 * 1024;

pub const DEFAULT_MODEL = "gpt-5.4-mini";
pub const DEFAULT_BASE_URL = "https://api.openai.com/v1";

pub const Client = struct {
    http: std.http.Client,
    api_key: []const u8,
    base_url: []const u8,
    dotenv_body: ?[]u8 = null,

    pub fn init(allocator: Allocator, io: Io, api_key: []const u8) Client {
        return .{
            .http = .{
                .allocator = allocator,
                .io = io,
            },
            .api_key = api_key,
            .base_url = DEFAULT_BASE_URL,
        };
    }

    pub fn fromEnv(allocator: Allocator, io: Io, environ_map: *EnvironMap) !Client {
        var body: ?[]u8 = null;
        errdefer if (body) |bytes| allocator.free(bytes);

        const env_api_key = normalizeEnvValue(environ_map.get("OPENAI_API_KEY"));
        const env_base_url = normalizeEnvValue(environ_map.get("OPENAI_BASE_URL"));

        if (env_api_key == null or env_base_url == null) {
            body = readDotEnvLocalAlloc(allocator, io) catch |err| switch (err) {
                error.FileNotFound => null,
                else => return err,
            };
        }

        const config = try resolveClientConfig(environ_map, body);

        var client = Client.init(allocator, io, config.api_key).withBaseUrl(config.base_url);
        client.dotenv_body = body;
        return client;
    }

    pub fn withBaseUrl(self: Client, base_url: []const u8) Client {
        var next = self;
        const trimmed = std.mem.trim(u8, base_url, "/");
        next.base_url = if (trimmed.len == 0) DEFAULT_BASE_URL else trimmed;
        return next;
    }

    pub fn deinit(self: *Client) void {
        if (self.dotenv_body) |body| self.http.allocator.free(body);
        self.http.deinit();
    }

    pub fn createResponse(self: *Client, allocator: Allocator, request: ResponsesRequest) !Response {
        if (request.stream orelse false) {
            return self.createResponseStreaming(allocator, request, .{});
        }

        const payload = try request.toJson(allocator);
        defer allocator.free(payload);

        const endpoint = try std.fmt.allocPrint(allocator, "{s}/responses", .{self.base_url});
        defer allocator.free(endpoint);

        const auth_header = try std.fmt.allocPrint(allocator, "Bearer {s}", .{self.api_key});
        defer allocator.free(auth_header);

        const uri = try std.Uri.parse(endpoint);
        var http_request = try self.http.request(.POST, uri, .{
            .headers = .{
                .authorization = .{ .override = auth_header },
                .content_type = .{ .override = "application/json" },
                .accept_encoding = .{ .override = "identity" },
            },
            .redirect_behavior = .unhandled,
        });
        defer http_request.deinit();

        http_request.transfer_encoding = .{ .content_length = payload.len };
        var body = try http_request.sendBodyUnflushed(&.{});
        try body.writer.writeAll(payload);
        try body.end();
        try http_request.connection.?.flush();

        var http_response = try http_request.receiveHead(&.{});
        const raw_body = try readResponseBodyAlloc(allocator, &http_response);
        errdefer allocator.free(raw_body);

        if (http_response.head.status != .ok) {
            std.log.err("OpenAI request failed: status={} body={s}", .{
                http_response.head.status,
                raw_body,
            });
            return error.OpenAIRequestFailed;
        }

        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw_body, .{});
        errdefer parsed.deinit();

        return .{
            .allocator = allocator,
            .body = raw_body,
            .parsed = parsed,
        };
    }

    pub fn createResponseStreaming(
        self: *Client,
        allocator: Allocator,
        request: ResponsesRequest,
        handler: StreamHandler,
    ) !Response {
        var streaming_request = request;
        streaming_request.stream = true;

        const payload = try streaming_request.toJson(allocator);
        defer allocator.free(payload);

        const endpoint = try std.fmt.allocPrint(allocator, "{s}/responses", .{self.base_url});
        defer allocator.free(endpoint);

        const auth_header = try std.fmt.allocPrint(allocator, "Bearer {s}", .{self.api_key});
        defer allocator.free(auth_header);

        const uri = try std.Uri.parse(endpoint);
        const extra_headers = [_]std.http.Header{
            .{ .name = "accept", .value = "text/event-stream" },
        };

        var http_request = try self.http.request(.POST, uri, .{
            .headers = .{
                .authorization = .{ .override = auth_header },
                .content_type = .{ .override = "application/json" },
                .accept_encoding = .{ .override = "identity" },
            },
            .extra_headers = &extra_headers,
            .redirect_behavior = .unhandled,
        });
        defer http_request.deinit();

        http_request.transfer_encoding = .{ .content_length = payload.len };
        var body = try http_request.sendBodyUnflushed(&.{});
        try body.writer.writeAll(payload);
        try body.end();
        try http_request.connection.?.flush();

        var http_response = try http_request.receiveHead(&.{});
        if (http_response.head.status != .ok) {
            const raw_body = try readResponseBodyAlloc(allocator, &http_response);
            defer allocator.free(raw_body);
            std.log.err("OpenAI request failed: status={} body={s}", .{
                http_response.head.status,
                raw_body,
            });
            return error.OpenAIRequestFailed;
        }

        return try readStreamedResponse(allocator, &http_response, handler);
    }
};

pub const ResponsesRequest = struct {
    model: []const u8,
    input: Input,
    instructions: ?[]const u8 = null,
    tools: []const Tool = &.{},
    tool_choice: ?ToolChoice = null,
    parallel_tool_calls: ?bool = null,
    previous_response_id: ?[]const u8 = null,
    store: ?bool = null,
    stream: ?bool = null,

    pub fn init(model: []const u8, input: Input) ResponsesRequest {
        return .{
            .model = model,
            .input = input,
        };
    }

    pub fn withInstructions(self: ResponsesRequest, instructions: []const u8) ResponsesRequest {
        var next = self;
        next.instructions = instructions;
        return next;
    }

    pub fn withTools(self: ResponsesRequest, tools: []const Tool) ResponsesRequest {
        var next = self;
        next.tools = tools;
        return next;
    }

    pub fn withToolChoice(self: ResponsesRequest, tool_choice: ToolChoice) ResponsesRequest {
        var next = self;
        next.tool_choice = tool_choice;
        return next;
    }

    pub fn withParallelToolCalls(self: ResponsesRequest, parallel_tool_calls: bool) ResponsesRequest {
        var next = self;
        next.parallel_tool_calls = parallel_tool_calls;
        return next;
    }

    pub fn withStore(self: ResponsesRequest, store: bool) ResponsesRequest {
        var next = self;
        next.store = store;
        return next;
    }

    pub fn withStream(self: ResponsesRequest, stream: bool) ResponsesRequest {
        var next = self;
        next.stream = stream;
        return next;
    }

    pub fn followup(
        self: ResponsesRequest,
        previous_response_id: []const u8,
        input_items: []const InputItem,
    ) ResponsesRequest {
        return .{
            .model = self.model,
            .input = .{ .items = input_items },
            .instructions = self.instructions,
            .tools = self.tools,
            .tool_choice = self.tool_choice,
            .parallel_tool_calls = self.parallel_tool_calls,
            .previous_response_id = previous_response_id,
            .store = self.store,
            .stream = self.stream,
        };
    }

    pub fn toJson(self: ResponsesRequest, allocator: Allocator) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(allocator);
        defer out.deinit();

        try std.json.Stringify.value(self, .{}, &out.writer);
        return try out.toOwnedSlice();
    }

    pub fn jsonStringify(self: ResponsesRequest, jws: anytype) !void {
        try jws.beginObject();
        try jws.objectField("model");
        try jws.write(self.model);
        try jws.objectField("input");
        try jws.write(self.input);

        if (self.instructions) |instructions| {
            try jws.objectField("instructions");
            try jws.write(instructions);
        }
        if (self.tools.len != 0) {
            try jws.objectField("tools");
            try jws.write(self.tools);
        }
        if (self.tool_choice) |tool_choice| {
            try jws.objectField("tool_choice");
            try jws.write(tool_choice);
        }
        if (self.parallel_tool_calls) |parallel_tool_calls| {
            try jws.objectField("parallel_tool_calls");
            try jws.write(parallel_tool_calls);
        }
        if (self.previous_response_id) |previous_response_id| {
            try jws.objectField("previous_response_id");
            try jws.write(previous_response_id);
        }
        if (self.store) |store| {
            try jws.objectField("store");
            try jws.write(store);
        }
        if (self.stream) |stream| {
            try jws.objectField("stream");
            try jws.write(stream);
        }

        try jws.endObject();
    }
};

pub const Input = union(enum) {
    text: []const u8,
    items: []const InputItem,

    pub fn jsonStringify(self: Input, jws: anytype) !void {
        switch (self) {
            .text => |text| try jws.write(text),
            .items => |items| try jws.write(items),
        }
    }
};

pub const InputItem = union(enum) {
    message: Message,
    message_text: struct {
        role: Role,
        text: []const u8,
    },
    function_call_output: FunctionCallOutput,
    raw_json: []const u8,

    pub fn userText(text: []const u8) InputItem {
        return .{
            .message_text = .{
                .role = .user,
                .text = text,
            },
        };
    }

    pub fn assistantText(text: []const u8) InputItem {
        return .{
            .message_text = .{
                .role = .assistant,
                .text = text,
            },
        };
    }

    pub fn functionCallOutput(call_id: []const u8, output: []const u8) InputItem {
        return .{
            .function_call_output = .{
                .call_id = call_id,
                .output = output,
            },
        };
    }

    pub fn rawJson(json: []const u8) InputItem {
        return .{ .raw_json = json };
    }

    pub fn jsonStringify(self: InputItem, jws: anytype) !void {
        switch (self) {
            .message => |message| {
                try jws.beginObject();
                try jws.objectField("type");
                try jws.write("message");
                try jws.objectField("role");
                try jws.write(message.role);
                try jws.objectField("content");
                try jws.write(message.content);
                try jws.endObject();
            },
            .message_text => |message| {
                try jws.beginObject();
                try jws.objectField("type");
                try jws.write("message");
                try jws.objectField("role");
                try jws.write(message.role);
                try jws.objectField("content");
                try jws.beginArray();
                if (message.role == .assistant) {
                    try jws.write(MessageInputContent.outputText(message.text));
                } else {
                    try jws.write(MessageInputContent.inputText(message.text));
                }
                try jws.endArray();
                try jws.endObject();
            },
            .function_call_output => |output| {
                try jws.beginObject();
                try jws.objectField("type");
                try jws.write("function_call_output");
                try jws.objectField("call_id");
                try jws.write(output.call_id);
                try jws.objectField("output");
                try jws.write(output.output);
                try jws.endObject();
            },
            .raw_json => |json| {
                try jws.beginWriteRaw();
                try jws.writer.writeAll(json);
                jws.endWriteRaw();
            },
        }
    }
};

pub const Message = struct {
    role: Role,
    content: []const MessageInputContent,
};

pub const FunctionCallOutput = struct {
    call_id: []const u8,
    output: []const u8,
};

pub const Role = enum {
    user,
    assistant,
    system,

    pub fn jsonStringify(self: Role, jws: anytype) !void {
        try jws.write(@tagName(self));
    }
};

pub const MessageInputContent = union(enum) {
    input_text: []const u8,
    output_text: []const u8,

    pub fn inputText(text: []const u8) MessageInputContent {
        return .{ .input_text = text };
    }

    pub fn outputText(text: []const u8) MessageInputContent {
        return .{ .output_text = text };
    }

    pub fn jsonStringify(self: MessageInputContent, jws: anytype) !void {
        switch (self) {
            .input_text => |text| {
                try jws.beginObject();
                try jws.objectField("type");
                try jws.write("input_text");
                try jws.objectField("text");
                try jws.write(text);
                try jws.endObject();
            },
            .output_text => |text| {
                try jws.beginObject();
                try jws.objectField("type");
                try jws.write("output_text");
                try jws.objectField("text");
                try jws.write(text);
                try jws.endObject();
            },
        }
    }
};

pub const Tool = struct {
    name: []const u8,
    parameters_json: []const u8,
    description: ?[]const u8 = null,
    strict: ?bool = null,

    pub fn function(name: []const u8, parameters_json: []const u8) Tool {
        return .{
            .name = name,
            .parameters_json = parameters_json,
        };
    }

    pub fn withDescription(self: Tool, description: []const u8) Tool {
        var next = self;
        next.description = description;
        return next;
    }

    pub fn withStrict(self: Tool, strict: bool) Tool {
        var next = self;
        next.strict = strict;
        return next;
    }

    pub fn jsonStringify(self: Tool, jws: anytype) !void {
        try jws.beginObject();
        try jws.objectField("type");
        try jws.write("function");
        try jws.objectField("name");
        try jws.write(self.name);
        if (self.description) |description| {
            try jws.objectField("description");
            try jws.write(description);
        }
        try jws.objectField("parameters");
        try jws.beginWriteRaw();
        try jws.writer.writeAll(self.parameters_json);
        jws.endWriteRaw();
        if (self.strict) |strict| {
            try jws.objectField("strict");
            try jws.write(strict);
        }
        try jws.endObject();
    }
};

pub const ToolChoice = enum {
    auto,
    required,
    none,

    pub fn jsonStringify(self: ToolChoice, jws: anytype) !void {
        try jws.write(@tagName(self));
    }
};

pub const OutputTextDeltaCallback = *const fn (
    context: ?*anyopaque,
    delta: []const u8,
) anyerror!void;

pub const ResponseCompletedCallback = *const fn (
    context: ?*anyopaque,
    response: *const Response,
) anyerror!void;

pub const StreamHandler = struct {
    context: ?*anyopaque = null,
    on_output_text_delta: ?OutputTextDeltaCallback = null,
    on_response_completed: ?ResponseCompletedCallback = null,

    pub fn emitTextDelta(self: StreamHandler, delta: []const u8) !void {
        if (self.on_output_text_delta) |callback| {
            try callback(self.context, delta);
        }
    }

    pub fn emitCompleted(self: StreamHandler, response: *const Response) !void {
        if (self.on_response_completed) |callback| {
            try callback(self.context, response);
        }
    }
};

pub const JsonBlob = struct {
    bytes: []u8,
};

pub const Response = struct {
    allocator: Allocator,
    body: []u8,
    parsed: std.json.Parsed(std.json.Value),

    pub fn deinit(self: *Response) void {
        self.parsed.deinit();
        self.allocator.free(self.body);
        self.* = undefined;
    }

    pub fn id(self: *const Response) ![]const u8 {
        const root = try self.rootObject();
        return try stringField(root, "id");
    }

    pub fn outputTextAlloc(self: *const Response, allocator: Allocator) ![]u8 {
        const output = try self.outputArray();
        var out: std.Io.Writer.Allocating = .init(allocator);
        defer out.deinit();

        for (output.items) |item| {
            const item_object = asObject(item) catch continue;
            const item_type = stringField(item_object, "type") catch continue;
            if (!std.mem.eql(u8, item_type, "message")) continue;

            const content_value = item_object.get("content") orelse continue;
            const content_array = asArray(content_value) catch continue;
            for (content_array.items) |content_item| {
                const content_object = asObject(content_item) catch continue;
                const content_type = stringField(content_object, "type") catch continue;
                if (!std.mem.eql(u8, content_type, "output_text")) continue;
                try out.writer.writeAll(try stringField(content_object, "text"));
            }
        }

        return try out.toOwnedSlice();
    }

    pub fn outputItemJsonBlobsAlloc(self: *const Response, allocator: Allocator) ![]JsonBlob {
        const output = try self.outputArray();
        var blobs: ArrayList(JsonBlob) = .empty;
        defer blobs.deinit(allocator);

        errdefer {
            for (blobs.items) |blob| allocator.free(blob.bytes);
        }

        for (output.items) |item| {
            var out: std.Io.Writer.Allocating = .init(allocator);
            defer out.deinit();

            try std.json.Stringify.value(item, .{}, &out.writer);
            try blobs.append(allocator, .{
                .bytes = try out.toOwnedSlice(),
            });
        }

        return try blobs.toOwnedSlice(allocator);
    }

    pub fn deinitJsonBlobs(allocator: Allocator, blobs: []JsonBlob) void {
        for (blobs) |blob| allocator.free(blob.bytes);
        allocator.free(blobs);
    }

    pub fn functionCallsAlloc(self: *const Response, allocator: Allocator) ![]FunctionCall {
        const output = try self.outputArray();
        var calls: ArrayList(FunctionCall) = .empty;
        defer calls.deinit(allocator);

        for (output.items) |item| {
            const item_object = asObject(item) catch continue;
            const item_type = stringField(item_object, "type") catch continue;
            if (!std.mem.eql(u8, item_type, "function_call")) continue;

            try calls.append(allocator, .{
                .id = optionalStringField(item_object, "id"),
                .call_id = try stringField(item_object, "call_id"),
                .name = try stringField(item_object, "name"),
                .arguments = try stringField(item_object, "arguments"),
            });
        }

        return try calls.toOwnedSlice(allocator);
    }

    fn rootObject(self: *const Response) !std.json.ObjectMap {
        return asObject(self.parsed.value);
    }

    fn outputArray(self: *const Response) !std.json.Array {
        const root = try self.rootObject();
        const output = root.get("output") orelse return error.InvalidOpenAIResponse;
        return asArray(output);
    }
};

pub const FunctionCall = struct {
    id: ?[]const u8,
    call_id: []const u8,
    name: []const u8,
    arguments: []const u8,

    pub fn argumentsJson(self: FunctionCall, allocator: Allocator) !std.json.Parsed(std.json.Value) {
        return std.json.parseFromSlice(std.json.Value, allocator, self.arguments, .{});
    }
};

pub const ToolCallHandler = *const fn (
    context: ?*anyopaque,
    arena: Allocator,
    call: FunctionCall,
) anyerror![]const u8;

pub fn runAgentLoop(
    allocator: Allocator,
    client: *Client,
    initial_request: ResponsesRequest,
    max_turns: usize,
    handler: ToolCallHandler,
    context: ?*anyopaque,
) !Response {
    var request = initial_request;
    var request_arena: ?std.heap.ArenaAllocator = null;
    defer if (request_arena) |*arena| arena.deinit();

    var turn: usize = 0;
    while (turn < max_turns) : (turn += 1) {
        var response = try client.createResponse(allocator, request);
        errdefer response.deinit();

        if (request_arena) |*arena| {
            arena.deinit();
            request_arena = null;
        }

        const calls = try response.functionCallsAlloc(allocator);
        defer allocator.free(calls);

        if (calls.len == 0) {
            return response;
        }

        var next_request_arena = std.heap.ArenaAllocator.init(allocator);
        errdefer next_request_arena.deinit();
        const next_alloc = next_request_arena.allocator();

        const previous_response_id = try next_alloc.dupe(u8, try response.id());
        const outputs = try next_alloc.alloc(InputItem, calls.len);

        for (calls, 0..) |call, i| {
            const output = try handler(context, next_alloc, call);
            outputs[i] = InputItem.functionCallOutput(
                try next_alloc.dupe(u8, call.call_id),
                output,
            );
        }

        response.deinit();
        request = request.followup(previous_response_id, outputs);
        request_arena = next_request_arena;
    }

    return error.AgentLoopExceededMaxTurns;
}

const ClientConfig = struct {
    api_key: []const u8,
    base_url: []const u8,
};

const DotEnvValues = struct {
    api_key: ?[]const u8 = null,
    base_url: ?[]const u8 = null,
};

const DotEnvEntry = struct {
    key: []const u8,
    value: []const u8,
};

fn resolveClientConfig(environ_map: *EnvironMap, body: ?[]const u8) !ClientConfig {
    const env_api_key = normalizeEnvValue(environ_map.get("OPENAI_API_KEY"));
    const env_base_url = normalizeEnvValue(environ_map.get("OPENAI_BASE_URL"));

    var file_api_key: ?[]const u8 = null;
    var file_base_url: ?[]const u8 = null;
    if (env_api_key == null or env_base_url == null) {
        if (body) |bytes| {
            const parsed = try parseDotEnv(bytes);
            file_api_key = normalizeEnvValue(parsed.api_key);
            file_base_url = normalizeEnvValue(parsed.base_url);
        }
    }

    return .{
        .api_key = env_api_key orelse file_api_key orelse return error.MissingOpenAIKey,
        .base_url = env_base_url orelse file_base_url orelse DEFAULT_BASE_URL,
    };
}

fn normalizeEnvValue(value: ?[]const u8) ?[]const u8 {
    const bytes = value orelse return null;
    const trimmed = std.mem.trim(u8, bytes, whitespace);
    if (trimmed.len == 0) return null;
    return trimmed;
}

fn readDotEnvLocalAlloc(allocator: Allocator, io: Io) ![]u8 {
    return Io.Dir.cwd().readFileAlloc(io, dotenv_path, allocator, .limited(dotenv_max_bytes));
}

fn parseDotEnv(body: []const u8) !DotEnvValues {
    var values: DotEnvValues = .{};
    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        const entry = try parseDotEnvLine(line) orelse continue;

        if (std.mem.eql(u8, entry.key, "OPENAI_API_KEY")) {
            values.api_key = entry.value;
        } else if (std.mem.eql(u8, entry.key, "OPENAI_BASE_URL")) {
            values.base_url = entry.value;
        }
    }

    return values;
}

fn parseDotEnvLine(raw_line: []const u8) !?DotEnvEntry {
    var line = std.mem.trim(u8, raw_line, whitespace);
    if (line.len == 0 or line[0] == '#') return null;

    if (std.mem.startsWith(u8, line, "export")) {
        if (line.len == "export".len or std.ascii.isWhitespace(line["export".len])) {
            line = std.mem.trimStart(u8, line["export".len..], whitespace);
        }
    }

    const eq_index = std.mem.indexOfScalar(u8, line, '=') orelse return null;
    const key = std.mem.trim(u8, line[0..eq_index], whitespace);
    if (key.len == 0) return null;

    return .{
        .key = key,
        .value = try parseDotEnvValue(line[eq_index + 1 ..]),
    };
}

fn parseDotEnvValue(raw_value: []const u8) ![]const u8 {
    var value = std.mem.trim(u8, raw_value, whitespace);
    if (value.len == 0) return value;

    const quote = value[0];
    if (quote == '"' or quote == '\'') {
        var i: usize = 1;
        while (i < value.len) : (i += 1) {
            if (value[i] != quote) continue;

            const trailing = std.mem.trim(u8, value[i + 1 ..], whitespace);
            if (trailing.len != 0 and trailing[0] != '#') return error.InvalidDotEnv;
            return value[1..i];
        }
        return error.InvalidDotEnv;
    }

    value = stripInlineComment(value);
    return std.mem.trimEnd(u8, value, whitespace);
}

fn stripInlineComment(value: []const u8) []const u8 {
    for (value, 0..) |byte, i| {
        if (byte != '#') continue;
        if (i == 0 or std.ascii.isWhitespace(value[i - 1])) {
            return value[0..i];
        }
    }
    return value;
}

fn readResponseBodyAlloc(
    allocator: Allocator,
    http_response: *std.http.Client.Response,
) ![]u8 {
    var response_body: std.Io.Writer.Allocating = .init(allocator);
    defer response_body.deinit();

    var transfer_buffer: [8192]u8 = undefined;
    const reader = http_response.reader(&transfer_buffer);
    _ = reader.streamRemaining(&response_body.writer) catch |err| switch (err) {
        error.ReadFailed => return http_response.bodyErr() orelse error.ReadFailed,
        else => |e| return e,
    };

    return try response_body.toOwnedSlice();
}

fn readStreamedResponse(
    allocator: Allocator,
    http_response: *std.http.Client.Response,
    handler: StreamHandler,
) !Response {
    var transfer_buffer: [8192]u8 = undefined;
    const reader = http_response.reader(&transfer_buffer);
    return readStreamedResponseFromReader(allocator, reader, handler) catch |err| switch (err) {
        error.ReadFailed => return http_response.bodyErr() orelse error.ReadFailed,
        else => |e| return e,
    };
}

fn readStreamedResponseFromReader(
    allocator: Allocator,
    reader: *std.Io.Reader,
    handler: StreamHandler,
) !Response {
    var event_name: ArrayList(u8) = .empty;
    defer event_name.deinit(allocator);

    var data: ArrayList(u8) = .empty;
    defer data.deinit(allocator);

    while (try reader.takeDelimiter('\n')) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        if (line.len == 0) {
            if (try dispatchSseEvent(allocator, event_name.items, data.items, handler)) |response| {
                return response;
            }
            event_name.clearRetainingCapacity();
            data.clearRetainingCapacity();
            continue;
        }

        if (line[0] == ':') continue;

        if (std.mem.startsWith(u8, line, "event:")) {
            event_name.clearRetainingCapacity();
            try event_name.appendSlice(allocator, sseFieldValue(line["event:".len..]));
            continue;
        }

        if (std.mem.startsWith(u8, line, "data:")) {
            if (data.items.len != 0) try data.append(allocator, '\n');
            try data.appendSlice(allocator, sseFieldValue(line["data:".len..]));
        }
    }

    if (try dispatchSseEvent(allocator, event_name.items, data.items, handler)) |response| {
        return response;
    }
    return error.OpenAIStreamEndedWithoutCompletion;
}

fn dispatchSseEvent(
    allocator: Allocator,
    event_name: []const u8,
    data: []const u8,
    handler: StreamHandler,
) !?Response {
    if (data.len == 0 or std.mem.eql(u8, data, "[DONE]")) return null;

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, data, .{});
    defer parsed.deinit();

    const object = try asObject(parsed.value);
    const event_type = if (event_name.len != 0)
        event_name
    else
        optionalStringField(object, "type") orelse return null;

    if (std.mem.eql(u8, event_type, "response.output_text.delta")) {
        const delta = try stringField(object, "delta");
        try handler.emitTextDelta(delta);
        return null;
    }

    if (std.mem.eql(u8, event_type, "response.completed")) {
        const response_value = object.get("response") orelse return error.InvalidOpenAIResponse;
        var response = try responseFromValue(allocator, response_value);
        errdefer response.deinit();
        try handler.emitCompleted(&response);
        return response;
    }

    if (std.mem.eql(u8, event_type, "response.failed")) {
        return error.OpenAIRequestFailed;
    }
    if (std.mem.eql(u8, event_type, "response.incomplete")) {
        return error.OpenAIResponseIncomplete;
    }

    return null;
}

fn responseFromValue(allocator: Allocator, value: std.json.Value) !Response {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();

    try std.json.Stringify.value(value, .{}, &out.writer);
    const body = try out.toOwnedSlice();
    errdefer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    errdefer parsed.deinit();

    return .{
        .allocator = allocator,
        .body = body,
        .parsed = parsed,
    };
}

fn sseFieldValue(value: []const u8) []const u8 {
    if (value.len != 0 and value[0] == ' ') return value[1..];
    return value;
}

fn asObject(value: std.json.Value) !std.json.ObjectMap {
    return switch (value) {
        .object => |object| object,
        else => error.InvalidOpenAIResponse,
    };
}

fn asArray(value: std.json.Value) !std.json.Array {
    return switch (value) {
        .array => |array| array,
        else => error.InvalidOpenAIResponse,
    };
}

fn asString(value: std.json.Value) ![]const u8 {
    return switch (value) {
        .string => |string| string,
        else => error.InvalidOpenAIResponse,
    };
}

fn stringField(object: std.json.ObjectMap, field_name: []const u8) ![]const u8 {
    const value = object.get(field_name) orelse return error.InvalidOpenAIResponse;
    return asString(value);
}

fn optionalStringField(object: std.json.ObjectMap, field_name: []const u8) ?[]const u8 {
    const value = object.get(field_name) orelse return null;
    return asString(value) catch null;
}

test "serializes flat function tool shape" {
    const allocator = std.testing.allocator;
    const request = ResponsesRequest.init("gpt-5", .{ .text = "hello" })
        .withTools(&.{
            Tool.function("get_weather",
                \\{
                \\  "type": "object",
                \\  "properties": {
                \\    "location": { "type": "string" },
                \\    "units": { "type": ["string", "null"], "enum": ["celsius", "fahrenheit"] }
                \\  },
                \\  "required": ["location", "units"],
                \\  "additionalProperties": false
                \\}
            )
                .withDescription("Look up weather")
                .withStrict(true),
        })
        .withParallelToolCalls(false)
        .withStore(true);

    const json = try request.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expectEqualStrings(
        "{\"model\":\"gpt-5\",\"input\":\"hello\",\"tools\":[{\"type\":\"function\",\"name\":\"get_weather\",\"description\":\"Look up weather\",\"parameters\":{\n  \"type\": \"object\",\n  \"properties\": {\n    \"location\": { \"type\": \"string\" },\n    \"units\": { \"type\": [\"string\", \"null\"], \"enum\": [\"celsius\", \"fahrenheit\"] }\n  },\n  \"required\": [\"location\", \"units\"],\n  \"additionalProperties\": false\n},\"strict\":true}],\"parallel_tool_calls\":false,\"store\":true}",
        json,
    );
}

test "serializes stream flag and raw input items" {
    const allocator = std.testing.allocator;
    const request = ResponsesRequest.init("gpt-5", .{
        .items = &.{
            InputItem.rawJson("{\"type\":\"reasoning\",\"id\":\"rs_123\"}"),
            InputItem.userText("hi"),
        },
    }).withStream(true);

    const json = try request.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expectEqualStrings(
        "{\"model\":\"gpt-5\",\"input\":[{\"type\":\"reasoning\",\"id\":\"rs_123\"},{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"hi\"}]}],\"stream\":true}",
        json,
    );
}

test "serializes assistant text as output text content" {
    const allocator = std.testing.allocator;
    const request = ResponsesRequest.init("gpt-5", .{
        .items = &.{
            InputItem.userText("hi"),
            InputItem.assistantText("hello"),
        },
    });

    const json = try request.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expectEqualStrings(
        "{\"model\":\"gpt-5\",\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"hi\"}]},{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\"}]}]}",
        json,
    );
}

test "followup uses function_call_output items" {
    const allocator = std.testing.allocator;
    const first = ResponsesRequest.init("gpt-5", .{
        .items = &.{InputItem.userText("hi")},
    }).withInstructions("be concise");

    const second = first.followup("resp_123", &.{
        InputItem.functionCallOutput("call_123", "{\"ok\":true}"),
    });

    const json = try second.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expectEqualStrings(
        "{\"model\":\"gpt-5\",\"input\":[{\"type\":\"function_call_output\",\"call_id\":\"call_123\",\"output\":\"{\\\"ok\\\":true}\"}],\"instructions\":\"be concise\",\"previous_response_id\":\"resp_123\"}",
        json,
    );
}

test "dotenv parsing and resolution prefer live env values" {
    const allocator = std.testing.allocator;
    const dotenv =
        \\# comment
        \\export OPENAI_API_KEY="sk-file"
        \\OPENAI_BASE_URL='https://dotenv.example/v1/' # trailing comment
        \\
    ;

    const parsed = try parseDotEnv(dotenv);
    try std.testing.expectEqualStrings("sk-file", parsed.api_key.?);
    try std.testing.expectEqualStrings("https://dotenv.example/v1/", parsed.base_url.?);

    var env = EnvironMap.init(allocator);
    defer env.deinit();
    try env.put("OPENAI_API_KEY", "  sk-env  ");

    const config = try resolveClientConfig(&env, dotenv);
    try std.testing.expectEqualStrings("sk-env", config.api_key);
    try std.testing.expectEqualStrings("https://dotenv.example/v1/", config.base_url);
}

test "dotenv resolution prefers environment over dotenv" {
    const allocator = std.testing.allocator;

    var env = EnvironMap.init(allocator);
    defer env.deinit();
    try env.put("OPENAI_API_KEY", "  sk-env  ");

    const config = try resolveClientConfig(&env,
        \\export OPENAI_API_KEY="sk-file"
        \\OPENAI_BASE_URL=https://dotenv.example/v1/ # trailing comment
    );
    try std.testing.expectEqualStrings("sk-env", config.api_key);
    try std.testing.expectEqualStrings("https://dotenv.example/v1/", config.base_url);
}

test "response output text joins output_text segments and extracts function calls" {
    const allocator = std.testing.allocator;
    var response = Response{
        .allocator = allocator,
        .body = try allocator.dupe(u8,
            \\{
            \\  "id": "resp_123",
            \\  "output": [
            \\    { "type": "reasoning" },
            \\    {
            \\      "type": "message",
            \\      "role": "assistant",
            \\      "content": [
            \\        { "type": "output_text", "text": "hello" },
            \\        { "type": "output_text", "text": " world" }
            \\      ]
            \\    },
            \\    {
            \\      "type": "function_call",
            \\      "call_id": "call_123",
            \\      "name": "noop",
            \\      "arguments": "{}"
            \\    }
            \\  ]
            \\}
        ),
        .parsed = undefined,
    };
    response.parsed = try std.json.parseFromSlice(std.json.Value, allocator, response.body, .{});
    defer response.deinit();

    const output_text = try response.outputTextAlloc(allocator);
    defer allocator.free(output_text);
    try std.testing.expectEqualStrings("hello world", output_text);

    const blobs = try response.outputItemJsonBlobsAlloc(allocator);
    defer Response.deinitJsonBlobs(allocator, blobs);
    try std.testing.expectEqual(@as(usize, 3), blobs.len);
    try std.testing.expectEqualStrings("{\"type\":\"reasoning\"}", blobs[0].bytes);

    const calls = try response.functionCallsAlloc(allocator);
    defer allocator.free(calls);
    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("noop", calls[0].name);
}

test "stream parser emits deltas and returns completed response" {
    const allocator = std.testing.allocator;
    var reader = Io.Reader.fixed(
        \\event: response.output_text.delta
        \\data: {"type":"response.output_text.delta","delta":"hel"}
        \\
        \\event: response.output_text.delta
        \\data: {"type":"response.output_text.delta","delta":"lo"}
        \\
        \\event: response.completed
        \\data: {"type":"response.completed","response":{"id":"resp_123","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}]}}
        \\
    );

    const Capture = struct {
        alloc: Allocator,
        deltas: ArrayList(u8) = .empty,
        completed: bool = false,
        completed_id: ?[]const u8 = null,

        fn onDelta(context: ?*anyopaque, delta: []const u8) !void {
            const self: *@This() = @ptrCast(@alignCast(context.?));
            try self.deltas.appendSlice(self.alloc, delta);
        }

        fn onCompleted(context: ?*anyopaque, response: *const Response) !void {
            const self: *@This() = @ptrCast(@alignCast(context.?));
            self.completed = true;
            self.completed_id = try response.id();
        }
    };

    var capture = Capture{
        .alloc = allocator,
    };
    defer capture.deltas.deinit(allocator);

    var response = try readStreamedResponseFromReader(allocator, &reader, .{
        .context = &capture,
        .on_output_text_delta = Capture.onDelta,
        .on_response_completed = Capture.onCompleted,
    });
    defer response.deinit();

    try std.testing.expectEqualStrings("hello", capture.deltas.items);
    try std.testing.expect(capture.completed);
    try std.testing.expectEqualStrings("resp_123", capture.completed_id.?);

    const output_text = try response.outputTextAlloc(allocator);
    defer allocator.free(output_text);
    try std.testing.expectEqualStrings("hello", output_text);
}
